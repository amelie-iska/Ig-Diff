"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""
import os
import time
from datetime import datetime
from typing import Dict, Optional

import GPUtil
import hydra
import numpy as np
import pandas as pd
import torch
import tree
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from analysis import utils as au
from data import utils as du
from experiments import train_se3_diffusion
from pathlib import Path


class Sampler:
    def __init__(self, conf: DictConfig, conf_overrides: Dict = None):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logger

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples
        self.batch_size = self._infer_conf.batch_size
        self.chain_break_jump = self._conf.data.chain_break_jump

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = "".join(
                    [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
                )
                self.device = f"cuda:{available_gpus[0]}"
                self._conf.experiment.num_gpus = 1
                self._conf.experiment.use_gpu = True
            else:
                self._conf.experiment.use_gpu = True
                self._conf.experiment.num_gpus = 1
                self.device = f"cuda:{self._infer_conf.gpu_id}"
        else:
            self._conf.experiment.use_gpu = False
            self._conf.experiment.num_gpus = 0
            self.device = "cpu"
        self._log.info(f"Using device: {self.device}")

        # Set-up directories
        self._weights_path = self._infer_conf.weights_path
        output_dir = self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        # print(output_dir, dt_string)
        self._output_dir = os.path.join(output_dir, dt_string)
        # print(self._output_dir)
        # self._output_dir = output_dir / dt_string
        Path(self._output_dir).mkdir(exist_ok=True, parents=True)
        self._log.info(f"Saving results to {self._output_dir}")
        config_path = Path(os.path.join(self._output_dir, "inference_conf.yaml"))
        # config_path = self._output_dir / "inference_conf.yaml"
        # print(config_path)
        with config_path.open("w") as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f"Saving inference config to {config_path}")

        # Load models and experiment
        self._load_ckpt(conf_overrides)

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f"Loading weights from {self._weights_path}")
        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True, map_location=self.device
        )
        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(self._conf.model, weights_pkl["conf"].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        # This is a horrible setup and I really want to change it but that would be a major refactor.
        # I really don't think that accessing the model and inference by initialising a training experiment
        # class makes much sense.
        # Stop the model from untarring the dataset that we don't need.
        self._conf.data.untarred = True
        self._conf.experiment.use_ddp = False
        self.exp = train_se3_diffusion.Experiment(conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl["model"]
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        csv_path = self._sample_conf.sample_length_csv
        sample_length_df = pd.read_csv(csv_path)
        for idx, row in sample_length_df.iterrows():
            chain_lengths_str = [L for L in row["chain_lengths"].split("|")]
            chain_lengths = [int(L) for L in chain_lengths_str]
            num_batches = row["num_batches"]
            length_dir = Path(self._output_dir) / f"length_{'_'.join(chain_lengths_str)}"
            length_dir.mkdir(exist_ok=True)
            self._log.info(
                f"Sampling {num_batches} batches for {len(chain_lengths)} chains of "
                f"length {' '.join(chain_lengths_str)} into {length_dir}"
            )
            for batch_i in range(num_batches):
                sample_output = self.sample(chain_lengths)
                for sample_i in range(self.batch_size):
                    sample_dir = os.path.join(
                        length_dir, f"sample_{batch_i}_{sample_i}"
                    )
                    os.makedirs(sample_dir, exist_ok=True)
                    traj_paths = self.save_traj(
                        bb_prot_traj=sample_output["prot_traj"][:, sample_i],
                        x0_traj=sample_output["rigid_0_traj"][:, sample_i],
                        diffuse_mask=np.ones(sum(chain_lengths)),
                        chain_idx=sample_output["chain_idx"][sample_i],
                        output_dir=sample_dir,
                    )

                    pdb_path = traj_paths["sample_path"]
                    self._log.info(f"Done sample {sample_i}: {pdb_path}")

    def save_traj(
        self,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        chain_idx: np.ndarray,
        output_dir: str,
    ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            chain_idx: which chain each residue belongs to.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, "sample")
        prot_traj_path = os.path.join(output_dir, "bb_traj")
        x0_traj_path = os.path.join(output_dir, "x0_traj")

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors,
            chain_idx=chain_idx,
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            chain_idx=chain_idx,
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj, x0_traj_path, b_factors=b_factors, chain_idx=chain_idx
        )
        return {
            "sample_path": sample_path,
            "traj_path": prot_traj_path,
            "x0_traj_path": x0_traj_path,
        }

    def sample(self, chain_lengths: list[int]):
        """Sample based on length.

        Args:
            chain_lengths: list of chain lengths
        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        total_length = sum(chain_lengths)
        res_mask = np.ones(total_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_samples = [
            self.diffuser.sample_ref(
                n_samples=total_length,
                as_tensor_7=True,
            )
            for _ in range(self.batch_size)
        ]

        # Create chain and residue index tensors accounting for chain breaks
        # based on chain lengths.
        res_idx_tensors = []
        chain_idx_tensors = []
        current_res_max = 0
        for i, chain_length in enumerate(chain_lengths):
            res_idx_chain = torch.arange(
                i * self.chain_break_jump + current_res_max,
                i * self.chain_break_jump + chain_length + current_res_max,
            )
            current_res_max = max(res_idx_chain)
            res_idx_tensors.append(res_idx_chain)
            chain_idx_tensors.append(torch.ones(chain_length) * i)
        res_idx = torch.cat(res_idx_tensors)
        chain_idx = torch.cat(chain_idx_tensors)

        init_feats = [
            {
                "res_mask": res_mask,
                "seq_idx": res_idx,
                "fixed_mask": fixed_mask,
                "chain_idx": chain_idx,
                "torsion_angles_sin_cos": np.zeros((total_length, 7, 2)),
                "sc_ca_t": np.zeros((total_length, 3)),
                **ref_sample,
            }
            for ref_sample in ref_samples
        ]
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        # Concatenate samples into a batch.
        init_feats = {
            key: torch.cat([d[key][None] for d in init_feats])
            for key in init_feats[0].keys()
        }
        init_feats = tree.map_structure(lambda x: x.to(self.device), init_feats)
        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        return sample_out


@hydra.main(
    version_base=None,
    config_path=str("../config/"),
    config_name="inference",
)
def main(conf: DictConfig) -> None:
    # # To avoid openMP issue on macos, set flag below as workaround to single OpenMP runtime constraint
    # # (note that this may cause crashes or silently produce incorrect results)
    # # https://github.com/dmlc/xgboost/issues/1715
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # Read model checkpoint.
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
