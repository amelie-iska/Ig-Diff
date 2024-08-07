"""Pytorch script for training SE(3) protein diffusion.

To run:

> python experiments/train_se3_diffusion.py

Without DVCLive,

> python experiments/train_se3_diffusion.py experiment.use_dvclive=False

To modify config options with the command line,

    > python experiments/train_se3_diffusion.py experiment.batch_size=32

"""
import copy
import os
import random
import shutil
import tarfile
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from typing import Union

import GPUtil
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from exs.dvckit import get_git_root, get_pipeline_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from analysis import metrics
from analysis import utils as au
from data import all_atom, pdb_data_loader, se3_diffuser
from data import utils as du
from experiments import utils as eu
from model import score_network
from openfold.utils import rigid_utils as ru


class Experiment:
    def __init__(
        self,
        *,
        conf: DictConfig,
    ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logger
        self._available_gpus = "".join(
            [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
        )

        # initialize ddp info if in ddp mode
        self._use_ddp = conf.experiment.use_ddp
        if self._use_ddp:
            dist.init_process_group(backend="nccl")
            self.ddp_info = eu.get_ddp_info()
            self.log(f"{self.ddp_info=}", rank_zero_only=False)
            self.log(
                f"{self._available_gpus=} {torch.cuda.device_count()=}",
                rank_zero_only=False,
                level="DEBUG",
            )
            torch.cuda.set_device(self.ddp_info["local_rank"])
        #
        # # Data processing
        # conf["data"]["csv_path"] = (
        #     get_pipeline_dir()
        #     / "data"
        #     / conf["data"]["dataset_dir"]
        #     / "metadata_clustered.csv"
        # )
        #
        # untarred_data_path = (
        #     get_pipeline_dir() / "data" / conf["data"]["dataset_dir"] / "untarred_data"
        # )
        # untarred_data_path.mkdir(exist_ok=True)
        # # For quick debugging, data is untarred manually prior to experiment.
        # if not conf["data"]["untarred"]:
        #     tarfile_path = (
        #         get_pipeline_dir()
        #         / "data"
        #         / conf["data"]["dataset_dir"]
        #         / "processed_pkls.tar.gz"
        #     )
        #     # Unpack data into dataset directory, only do this on master node if using DDP.
        #     if not self._use_ddp or (self._use_ddp and self.ddp_info["rank"] == 0):
        #         with tarfile.open(tarfile_path) as tarfile_obj:
        #             tarfile_obj.extractall(untarred_data_path)
        #     if self._use_ddp:
        #         dist.barrier()
        #
        # conf["data"]["data_root"] = str(untarred_data_path)

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        # self._use_dvclive = conf.experiment.use_dvclive
        # self._data_conf = conf.data
        #
        # if self._use_dvclive and (
        #     not self._use_ddp or (self._use_ddp and self.ddp_info["rank"] == 0)
        # ):
        #     self._live = Live(dir=str(get_pipeline_dir() / "metrics"), dvcyaml=False)
        # else:
        #     self._live = None

        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if conf.experiment.warm_start:
            self.log(f"Warm starting from: {conf.experiment.warm_start}")
            if (
                ".pkl" not in conf.experiment.warm_start
                and ".pth" not in conf.experiment.warm_start
            ):
                raise ValueError(
                    f"Extension for {conf.experiment.warm_start} is not .pkl or .pth."
                )
            ckpt_path = "./" / conf.experiment.warm_start
            self.log(f"Loading checkpoint from {ckpt_path}", rank_zero_only=False)
            if self._exp_conf.use_gpu:
                ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            else:
                ckpt_pkl = du.read_pkl(
                    ckpt_path, use_torch=True, map_location=torch.device("cpu")
                )
            ckpt_model = ckpt_pkl["model"]
            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl["conf"])
                OmegaConf.set_struct(conf, True)

            # For compatibility with older checkpoints.
            if "optimizer" in ckpt_pkl:
                ckpt_opt = ckpt_pkl["optimizer"]
            if "epoch" in ckpt_pkl and self._exp_conf.load_optimizer_state:
                self.trained_epochs = ckpt_pkl["epoch"]
            if "step" in ckpt_pkl and self._exp_conf.load_optimizer_state:
                self.trained_steps = ckpt_pkl["step"]

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = score_network.ScoreNetwork(self._model_conf, self.diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace("module.", ""): v for k, v in ckpt_model.items()}
            incompat_keys = self._model.load_state_dict(ckpt_model, strict=False)
            if incompat_keys.missing_keys:
                if conf["model"][
                    "use_chain_embedding"
                ] and incompat_keys.missing_keys == [
                    "embedding_layer.chain_embedder.weight",
                    "embedding_layer.chain_embedder.bias",
                ]:
                    warnings.warn(
                        f"Got missing keys {incompat_keys.missing_keys=} during ckpt loading, "
                        "these layers shall be left with randomnly initialised weights."
                    )
                else:
                    raise ValueError(
                        f"Got missing keys {incompat_keys.missing_keys=} during ckpt loading."
                    )
            if incompat_keys.unexpected_keys:
                raise ValueError(
                    f"Got unexpected keys {incompat_keys.unexpected_keys=} during ckpt loading."
                )
        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self.log(f"Number of model parameters {num_parameters}")
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate
        )
        if ckpt_opt is not None and self._exp_conf.load_optimizer_state:
            self._optimizer.load_state_dict(ckpt_opt)

        # Set-up checkpoint location
        ckpt_dir = "./checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = str(ckpt_dir)
        self.log(f"Checkpoints saved to: {ckpt_dir}")
        eval_dir = get_pipeline_dir() / "metrics" / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._exp_conf.eval_dir = str(eval_dir)
        self.log(f"Evaluation saved to: {eval_dir}")

        self._aux_data_history = deque(maxlen=100)

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):

        # Datasets
        train_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf, diffuser=self._diffuser, is_training=True
        )

        valid_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf, diffuser=self._diffuser, is_training=False
        )
        if self._use_ddp:
            train_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                length_batch_uniform=self._exp_conf.length_batch_uniform,
                rank=self.ddp_info["rank"],
            )
        else:
            train_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                length_batch_uniform=self._exp_conf.length_batch_uniform,
            )

        valid_sampler = None

        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=(
                self._exp_conf.batch_size
                if not self._exp_conf.use_ddp
                else self._exp_conf.batch_size // self.ddp_info["world_size"]
            ),
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        replica_id = 0
        if self._use_ddp and not self._exp_conf.use_gpu:
            raise ValueError(
                "use_gpu must be set to True in experimental configuration if using DDP."
            )

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus == 1:
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self.log(f"Using device: {device}", rank_zero_only=False)
            # muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [
                    f"cuda:{i}" for i in self._available_gpus[: self._exp_conf.num_gpus]
                ]
                # DDP mode
                if self._use_ddp:
                    device = torch.device("cuda", self.ddp_info["local_rank"])
                    model = self.model.to(device)
                    self._model = DDP(
                        model,
                        device_ids=[self.ddp_info["local_rank"]],
                        output_device=self.ddp_info["local_rank"],
                        find_unused_parameters=True,
                    )
                    self.log(
                        f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']},"
                        f" devices: {device_ids}",
                        rank_zero_only=False,
                    )
                # DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(
                            f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available "
                        )
                    self.log(
                        f"Multi-GPU training on GPUs in DP mode: {device_ids}",
                        rank_zero_only=False,
                    )
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
            else:
                raise ValueError(
                    f"Got {self._exp_conf.num_gpus=} but num_gpus must be an integer larger than 0."
                )
        else:
            device = "cpu"
            self._model = self.model.to(device)
        self.log(f"Using device: {device}", rank_zero_only=False)

        self._model.train()

        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader, valid_loader, device, return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self.log("Done", rank_zero_only=False)
        ckpt_path = os.path.join(self.ckpt_dir, f"step_{self.trained_steps}.pth")
        du.write_checkpoint(
            ckpt_path,
            copy.deepcopy(self.model.state_dict()),
            self._conf,
            copy.deepcopy(self._optimizer.state_dict()),
            self.trained_epochs,
            self.trained_steps,
            logger=self._log,
            use_torch=True,
        )
        # if self._live is not None:
        #     self._live.end()
        #     self._live = None
        # if not self._data_conf["untarred"]:
        #     if not self._use_ddp or (self._use_ddp and self.ddp_info["rank"] == 0):
        #         shutil.rmtree(self._data_conf["data_root"])
        #     if self._use_ddp:
        #         dist.barrier()
        if not self._use_ddp or (self._use_ddp and self.ddp_info["rank"] == 0):
            ckpt_path = os.path.join(self.ckpt_dir, f"step_{self.trained_steps}.pth")
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True,
            )
        if self._use_ddp:
            dist.barrier()
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        for train_feats in train_loader:
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k, v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Take checkpoint
            if (self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (
                self._exp_conf.early_ckpt and self.trained_steps == 100
            ):
                if not self._use_ddp or (self._use_ddp and self.ddp_info["rank"] == 0):
                    ckpt_path = os.path.join(
                        self.ckpt_dir, f"step_{self.trained_steps}.pth"
                    )
                    du.write_checkpoint(
                        ckpt_path,
                        copy.deepcopy(self.model.state_dict()),
                        self._conf,
                        copy.deepcopy(self._optimizer.state_dict()),
                        self.trained_epochs,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True,
                    )

                    # Run evaluation
                    self.log(f"Running evaluation of {ckpt_path}", rank_zero_only=False)
                    start_time = time.time()
                    eval_dir = os.path.join(
                        self._exp_conf.eval_dir, f"step_{self.trained_steps}"
                    )
                    os.makedirs(eval_dir, exist_ok=True)
                    ckpt_metrics = self.eval_fn(
                        eval_dir,
                        valid_loader,
                        device,
                        noise_scale=self._exp_conf.noise_scale,
                    )
                    eval_time = time.time() - start_time
                    self.log(
                        f"Finished evaluation in {eval_time:.2f}s", rank_zero_only=False
                    )
                else:
                    ckpt_metrics = None
                    eval_time = None
            else:
                ckpt_metrics = None
                eval_time = None

            if (
                self.trained_steps == 1
                or self.trained_steps % self._exp_conf.log_freq == 0
            ):
                # Logging to terminal

                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = (
                    min(self._exp_conf.log_freq, self.trained_steps) / elapsed_time
                )
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = " ".join(
                    [
                        f"{k}={v[0]:.4f}"
                        for k, v in rolling_losses.items()
                        if "batch" not in k
                    ]
                )
                self.log(
                    f"[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}",
                    rank_zero_only=False,
                )
                log_lossses = defaultdict(list)
                #
                # # Remote log to DVCLive.
                # if self._use_dvclive:
                #     if self._live:
                #         self._live.step = self.trained_steps
                #     step_time = elapsed_time / min(
                #         self._exp_conf.log_freq, self.trained_steps
                #     )
                #     example_per_sec = self._exp_conf.batch_size / step_time
                #
                #     self.log_live_metric("loss", loss.item(), reduce=True)
                #     self.log_live_metric("epochs", self.trained_epochs)
                #     self.log_live_metric(
                #         "examples_total", self._exp_conf.batch_size * self.trained_steps
                #     )
                #     self.log_live_metric(
                #         "example_per_sec", example_per_sec, reduce=True
                #     )
                #     self.log_live_metric(
                #         "rotation_loss", aux_data["rot_loss"].item(), reduce=True
                #     )
                #     self.log_live_metric(
                #         "translation_loss", aux_data["trans_loss"].item(), reduce=True
                #     )
                #     self.log_live_metric(
                #         "bb_atom_loss", aux_data["bb_atom_loss"].item(), reduce=True
                #     )
                #     # self.log_live_metric("dist_mat_loss", aux_data["batch_dist_mat_loss"].item())
                #     self.log_live_metric(
                #         "local_batch_size", aux_data["examples_per_step"].item()
                #     )
                #     self.log_live_metric(
                #         "res_length", aux_data["res_length"].item(), reduce=True
                #     )
                #     strat_loss, strat_count = eu.t_stratified_loss(
                #         du.move_to_np(train_feats["t"]),
                #         du.move_to_np(aux_data["batch_rot_loss"]),
                #         loss_name="rot_loss",
                #     )
                #     self.log_live_strat_metric(strat_loss, strat_count, reduce=True)
                #
                #     strat_loss, strat_count = eu.t_stratified_loss(
                #         du.move_to_np(train_feats["t"]),
                #         du.move_to_np(aux_data["batch_trans_loss"]),
                #         loss_name="trans_loss",
                #     )
                #     self.log_live_strat_metric(strat_loss, strat_count, reduce=True)
                #
                #     strat_loss, strat_count = eu.t_stratified_loss(
                #         du.move_to_np(train_feats["t"]),
                #         du.move_to_np(aux_data["batch_bb_atom_loss"]),
                #         loss_name="bb_atom_loss",
                #     )
                #     self.log_live_strat_metric(strat_loss, strat_count, reduce=True)
                #
                #     strat_loss, strat_count = eu.t_stratified_loss(
                #         du.move_to_np(train_feats["t"]),
                #         du.move_to_np(aux_data["batch_dist_mat_loss"]),
                #         loss_name="dist_mat_loss",
                #     )
                #     self.log_live_strat_metric(strat_loss, strat_count)
                #
                #     if ckpt_metrics is not None:
                #         self.log_live_metric("eval_time", eval_time)
                #         for metric_name in metrics.ALL_METRICS:
                #             self.log_live_metric(
                #                 metric_name, ckpt_metrics[metric_name].mean()
                #             )

            if torch.isnan(loss):
                raise Exception(f"NaN encountered")

        if return_logs:
            return global_logs

    def eval_fn(
        self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0
    ):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats["res_mask"].bool())
            fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
            aatype = du.move_to_np(valid_feats["aatype"])
            chain_idx = du.move_to_np(valid_feats["chain_idx"])
            gt_prot = du.move_to_np(valid_feats["atom37_pos"])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats, min_t=min_t, num_t=num_t, noise_scale=noise_scale
            )
            final_prot = infer_out["prot_traj"][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_diffused_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_diffused = np.sum(unpad_diffused_mask) / num_res

                # Extract argmax predicted aatype
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    os.path.join(
                        eval_dir,
                        f"len_{num_res}_sample_{i}_diffused_{percent_diffused:.2f}.pdb",
                    ),
                    chain_idx=chain_idx[i],
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        chain_idx=chain_idx[i],
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        diffuse_mask=unpad_diffused_mask,
                    )
                except ValueError as e:
                    self.log(
                        f"Failed evaluation of length {num_res} sample {i}: {e}",
                        rank_zero_only=False,
                        level="WARNING",
                    )
                    continue
                sample_metrics["step"] = self.trained_steps
                sample_metrics["num_res"] = num_res
                sample_metrics["fixed_residues"] = np.sum(unpad_fixed_mask)
                sample_metrics["diffused_percentage"] = percent_diffused
                sample_metrics["sample_path"] = saved_path
                sample_metrics["gt_pdb"] = pdb_names[i]
                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, "metrics.csv")
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch)
        bb_mask = batch["res_mask"]
        diffuse_mask = 1 - batch["fixed_mask"]
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch["rot_score"]
        gt_trans_score = batch["trans_score"]
        rot_score_scaling = batch["rot_score_scaling"]
        trans_score_scaling = batch["trans_score_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out["rot_score"] * diffuse_mask[..., None]
        pred_trans_score = model_out["trans_score"] * diffuse_mask[..., None]

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score) ** 2 * loss_mask[
            ..., None
        ]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None] ** 2, dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out["rigids"][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_score_loss * (
            batch["t"] > self._exp_conf.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self._exp_conf.trans_x0_threshold)
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._diff_conf.diffuse_trans)

        # Rotation loss
        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._diff_conf.diffuse_rot)

        # Backbone atom loss
        pred_atom37 = model_out["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self._exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append(
            {"aux_data": aux_data, "model_out": model_out, "batch": batch}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1 / 2 * beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats["rot_score_scaling"] = rot_score_scaling * t_placeholder
        feats["trans_score_scaling"] = trans_score_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1 / num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_score = model_out["rot_score"]
                    trans_score = model_out["trans_score"]
                    rigid_pred = model_out["rigids"]
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                    diffuse_mask = (1 - sample_feats["fixed_mask"]) * sample_feats[
                        "res_mask"
                    ]
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
                else:
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out["rigids"])
                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    diffuse_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
            "chain_idx": du.move_to_np(sample_feats["chain_idx"]),
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret
    #
    # def log_live_metric(
    #     self,
    #     name: str,
    #     metric: Union[int, float, str],
    #     reduce: bool = False,
    #     reduce_op: str = "mean",
    # ) -> None:
    #     """
    #     Wrapper for using dvclive logging with DDP (multiprocessing).
    #     Args:
    #         name: passed to live.log_metric
    #         metric: passed to live.log_metric
    #         reduce: Whether to perform a reduction onto the rank 0 node of the metric prior to logging.
    #         reduce_op: Which operation to perform upon reduce, default is mean, options are min, max, mean and sum.
    #
    #     Returns:
    #
    #     """
    #     # Log normally if we aren't in multiproccessing
    #     if not self._use_ddp:
    #         self._live.log_metric(name, metric)
    #     else:
    #         if reduce:
    #             # Reduce only works on tensors:
    #             if isinstance(metric, torch.Tensor):
    #                 metric_tensor = metric
    #             else:
    #                 device = torch.device(f"cuda:{self.ddp_info['local_rank']}")
    #                 metric_tensor = torch.Tensor([metric]).to(device)
    #             # Perform reduction based on reduce operation:
    #             if reduce_op in ["mean", "sum"]:
    #                 dist.reduce(metric_tensor, dst=0)
    #                 if reduce_op == "mean":
    #                     if self.ddp_info["rank"] == 0:
    #                         metric_tensor = metric_tensor / dist.get_world_size()
    #             elif reduce_op == "min":
    #                 dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.MIN)
    #             elif reduce_op == "max":
    #                 dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.MAX)
    #             else:
    #                 raise ValueError(
    #                     f"Did not recognise {reduce_op=}, should be one of 'min', 'max', 'mean' or 'sum'"
    #                 )
    #             # Log the metric on master process
    #             if self.ddp_info["rank"] == 0:
    #                 self._live.log_metric(name, metric_tensor.item())
    #         # Log the metric on master process
    #         elif self.ddp_info["rank"] == 0:
    #             self._live.log_metric(name, metric)
    #
    # def log_live_strat_metric(
    #     self,
    #     metric_dict: dict[str, float],
    #     metric_count_dict: dict[str, int],
    #     reduce: bool = False,
    # ) -> None:
    #     """
    #     Wrapper for using dvclive logging with DDP (multiprocessing). We need to handle stratefied metrics seperately
    #     as we need to compute the mean per bin.
    #     Args:
    #         metric_dict: stratified loss dictionary to be logged
    #         metric_count_dict: bin counts for stratified loss.
    #         reduce: Whether to perform a reduction onto the rank 0 node of the metric prior to logging.
    #     Returns:
    #
    #     """
    #     # Log normally if we aren't in multiproccessing
    #     if not self._use_ddp:
    #         for l in metric_dict:
    #             if metric_count_dict[l] > 0:
    #                 self._live.log_metric(l, metric_dict[l] / metric_count_dict[l])
    #
    #     else:
    #         if reduce:
    #             # Reduce only works on tensors:
    #             device = torch.device(f"cuda:{self.ddp_info['local_rank']}")
    #             metric_tensor = torch.Tensor(list(metric_dict.values())).to(device)
    #             metric_counts_tensor = torch.Tensor(
    #                 list(metric_count_dict.values())
    #             ).to(device)
    #
    #             # Perform reduction based on sumation operation:
    #             dist.reduce(metric_tensor, dst=0)
    #             dist.reduce(metric_counts_tensor, dst=0)
    #             # Log the metric on master process
    #             if self.ddp_info["rank"] == 0:
    #                 keys_list = list(metric_dict.keys())
    #                 for i, (metric, metric_count) in enumerate(
    #                     zip(metric_tensor, metric_counts_tensor)
    #                 ):
    #                     if metric_count.item() > 0:
    #                         self._live.log_metric(
    #                             keys_list[i], (metric / metric_count).item()
    #                         )
    #
    #         # Log the metric on master process
    #         elif self.ddp_info["rank"] == 0:
    #             for l in metric_dict:
    #                 if metric_count_dict[l] > 0:
    #                     self._live.log_metric(l, metric_dict[l] / metric_count_dict[l])
    #
    # def log(
    #     self, message: str, rank_zero_only: bool = True, level: str = "INFO"
    # ) -> None:
    #     """
    #     Wrapper for writing out logs that handles using DDP. If using DDP, if rank_zero_only is true,
    #     only logs from rank zero. Otherwise it prepends a string describing the rank.
    #
    #     Args:
    #         message: Text to be logged.
    #         rank_zero_only: Write log from rank zero only in DDP mode.
    #         level: Level at which to log, passed to logger. Default is INFO.
    #     """
    #     if not self._use_ddp:
    #         self._log.log(level, message)
    #     else:
    #         if rank_zero_only:
    #             if self.ddp_info["rank"] == 0:
    #                 self._log.log(level, message)
    #         else:
    #             self._log.log(level, f"rank = {self.ddp_info['rank']} : {message}")


@hydra.main(
    version_base=None,
    config_path=str("../config/"),
    config_name="training_main",
)
def main(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    # os.environ["WANDB_START_METHOD"] = "thread"

    # # To avoid openMP issue on macos, set flag below as workaround to single OpenMP runtime constraint
    # # (note that this may cause crashes or silently produce incorrect results)
    # # https://github.com/dmlc/xgboost/issues/1715
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == "__main__":
    main()
