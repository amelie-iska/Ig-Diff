"""PDB dataset loader."""
import functools as fn
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from loguru import logger
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils import data

from data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils


def _rog_quantile_curve(df: pd.DataFrame, quantile: float, eval_x: np.ndarray) -> Any:
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y


class PdbDataset(data.Dataset):
    def __init__(
        self,
        *,
        data_conf: DictConfig,
        diffuser,
        is_training: bool,
    ) -> None:
        self._log = logger
        self._is_training = is_training
        self._data_conf = data_conf
        self._csv_path = Path(self.data_conf.csv_path)
        self._data_root = Path(self.data_conf.data_root)
        self._init_metadata()
        self._diffuser = diffuser
        self._chain_break_jump = data_conf["chain_break_jump"]

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def chain_break_jump(self):
        return self._chain_break_jump

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self._csv_path)
        self.raw_csv = pdb_csv
        if (
            filter_conf.allowed_oligomer is not None
            and len(filter_conf.allowed_oligomer) > 0
        ):
            pdb_csv = pdb_csv[
                pdb_csv.oligomeric_detail.isin(filter_conf.allowed_oligomer)
            ]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile is not None and filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, filter_conf.rog_quantile, np.arange(filter_conf.max_len)
            )
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x - 1]
            )
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[: filter_conf.subset]
        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )

    # cache make the sample in the same batch
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path: Path) -> dict[str, Any]:
        processed_feats = du.read_pkl(str(processed_file_path))
        processed_feats = du.center_and_scale_coords(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats["modeled_idx"]
        processed_feats = tree.map_structure(
            lambda x: x[min_idx : (max_idx + 1)], processed_feats
        )

        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": torch.tensor(processed_feats["aatype"]).long(),
            "all_atom_positions": torch.tensor(
                processed_feats["atom_positions"]
            ).double(),
            "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Change residue idx so that it increases across increasing chain idx.
        # Also add a jump in residue across chain breaks..
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats["residue_index"]
        new_res_idx = np.zeros_like(res_idx)
        all_chain_ids = sorted(np.unique(chain_idx))

        # Here we loop over all chain ids, which are unique.
        # We obtain a mask for where the chain_idx matches the chain id in question.
        # We then increment each residue idx in that chain by the previous maximum residue index
        # plus a jump for any breaks in chain.
        starting_res_idx = 0
        for i, chain_id in enumerate(all_chain_ids):
            chain_mask = (chain_idx == chain_id).astype(int)
            new_res_idx = new_res_idx + (res_idx + starting_res_idx) * chain_mask
            starting_res_idx = np.max(new_res_idx) + self.chain_break_jump

        # To speed up processing, only take necessary features
        final_feats = {
            "aatype": chain_feats["aatype"],
            "seq_idx": new_res_idx,
            "chain_idx": chain_idx,
            "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
            "residue_index": processed_feats["residue_index"],
            "res_mask": processed_feats["bb_mask"],
            "atom37_pos": chain_feats["all_atom_positions"],
            "atom37_mask": chain_feats["all_atom_mask"],
            "atom14_pos": chain_feats["atom14_gt_positions"],
            "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
        }
        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order["CA"]]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min,
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min, high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(f"Unable to generate diffusion mask for {row}")
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if "pdb_name" in csv_row:
            pdb_name = csv_row["pdb_name"]
        elif "chain_name" in csv_row:
            pdb_name = csv_row["chain_name"]
        else:
            raise ValueError("Need chain identifier.")
        processed_file_path = self._data_root / csv_row["processed_path"]
        chain_feats = self._process_csv_row(processed_file_path)

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[
            :, 0
        ]
        diffused_mask = np.ones_like(chain_feats["res_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        chain_feats["fixed_mask"] = fixed_mask
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid, t=t, diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_seq_len"])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class TrainSampler(data.Sampler):
    def __init__(
        self,
        *,
        data_conf: DictConfig,
        dataset: PdbDataset,
        batch_size: int,
        sample_mode: str,
        length_batch_uniform: bool,
    ):
        self._log = logger
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self._length_batch_uniform = length_batch_uniform

        if sample_mode == "length_batch":
            if length_batch_uniform:
                self.sampler_len = (
                    len(self._data_csv["modeled_seq_len"].unique()) * self._batch_size
                )
            else:
                counts = self._data_csv["modeled_seq_len"].value_counts().to_numpy()
                # We oversample each length to create a full batch if not divisible by batch_size.
                epoch_size = np.sum(
                    np.ceil(counts / self._batch_size) * self._batch_size
                )
                self.sampler_len = int(epoch_size)
        elif sample_mode == "time_batch":
            self.sampler_len = len(self._dataset_indices) * self._batch_size
        elif self._sample_mode in ["cluster_length_batch", "cluster_time_batch"]:
            self._pdb_to_cluster = self._read_clusters()
            num_clusters = len(set(self._data_csv["cluster_ids"]))
            self._log.info(f"Training on {num_clusters} clusters.")
            if self._sample_mode == "cluster_time_batch":
                self.sampler_len = num_clusters * self._batch_size
            else:
                sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                    1, random_state=0
                )
                if length_batch_uniform:
                    self.sampler_len = (
                        len(sampled_clusters["modeled_seq_len"].unique())
                        * self._batch_size
                    )
                else:
                    counts = (
                        sampled_clusters["modeled_seq_len"].value_counts().to_numpy()
                    )
                    # We oversample each length to create a full batch if not divisible by batch_size.
                    epoch_size = np.sum(
                        np.ceil(counts / self._batch_size) * self._batch_size
                    )
                    self.sampler_len = int(epoch_size)
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")

    def _read_clusters(self):
        pdb_to_cluster = {
            row.pdb_name: row.cluster_ids for row in self._data_csv.itertuples()
        }
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == "length_batch":
            # Each batch contains multiple proteins of the same length.
            # Make 1 batch for each length.
            if self._length_batch_uniform:
                sampled_order = self._data_csv.groupby("modeled_seq_len").sample(
                    self._batch_size, replace=True, random_state=self.epoch
                )
                # Sampler length may change!
                self.sampler_len = len(sampled_order)
                return iter(sampled_order["index"].tolist())
            # Use all training examples for each length then sample with replacement to complete the batch if necessary.
            else:
                sampled_order = []
                for _, group_elem in self._data_csv.groupby("modeled_seq_len"):
                    group_size = len(group_elem)
                    full_sample = []
                    exhaustive_sample = group_elem.sample(
                        frac=1, random_state=self.epoch
                    )["index"].tolist()
                    full_sample += exhaustive_sample
                    remainder = (
                        self._batch_size - group_size % self._batch_size
                    ) % self._batch_size
                    if remainder > 0:
                        remainder_sample = group_elem.sample(
                            n=remainder, replace=True, random_state=self.epoch
                        )["index"].tolist()
                        full_sample += remainder_sample
                    sampled_order += full_sample
                # Sampler length may change!
                self.sampler_len = len(sampled_order)
                return iter(sampled_order)
        elif self._sample_mode == "time_batch":
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == "cluster_length_batch":
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                1, random_state=self.epoch
            )
            # Sample 1 batch from each length with replacement.
            if self._length_batch_uniform:
                sampled_order = sampled_clusters.groupby("modeled_seq_len").sample(
                    self._batch_size, replace=True, random_state=self.epoch
                )
                # Sampler length may change!
                self.sampler_len = len(sampled_order)
                return iter(sampled_order["index"].tolist())
            # Use all training examples for each length then sample with replacement to complete the batch if necessary.
            else:
                sampled_order = []
                for _, group_elem in sampled_clusters.groupby("modeled_seq_len"):
                    group_size = len(group_elem)
                    full_sample = []
                    exhaustive_sample = group_elem.sample(
                        frac=1, random_state=self.epoch
                    )["index"].tolist()
                    full_sample += exhaustive_sample
                    remainder = (
                        self._batch_size - group_size % self._batch_size
                    ) % self._batch_size
                    if remainder > 0:
                        remainder_sample = group_elem.sample(
                            n=remainder, replace=True, random_state=self.epoch
                        )["index"].tolist()
                        full_sample += remainder_sample
                    sampled_order += full_sample
                # Sampler length may change!
                self.sampler_len = len(sampled_order)
                return iter(sampled_order)
        elif self._sample_mode == "cluster_time_batch":
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                1, random_state=self.epoch
            )
            dataset_indices = sampled_clusters["index"].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len


# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        *,
        data_conf: DictConfig,
        dataset: PdbDataset,
        batch_size: int,
        sample_mode: str,
        length_batch_uniform: bool,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self._data_conf = data_conf
        self._dataset = dataset
        self._log = logger
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        self._sample_mode = sample_mode
        self._length_batch_uniform = length_batch_uniform
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            seed = self.seed + self.epoch
        else:
            seed = self.seed

        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        if sample_mode == "length_batch":
            if length_batch_uniform:
                self._repeated_size = (
                    len(self._data_csv["modeled_seq_len"].unique()) * self._batch_size
                )
            else:
                counts = self._data_csv["modeled_seq_len"].value_counts().to_numpy()
                # We oversample each length to create a full batch if not divisible by batch_size.
                epoch_size = np.sum(
                    np.ceil(counts / self._batch_size) * self._batch_size
                )
                self._repeated_size = int(epoch_size)
        elif sample_mode == "time_batch":
            self._repeated_size = len(self._dataset_indices) * self._batch_size
        elif self._sample_mode in ["cluster_length_batch", "cluster_time_batch"]:
            self._pdb_to_cluster = self._read_clusters()
            num_clusters = len(set(self._data_csv["cluster_ids"]))
            if self._sample_mode == "cluster_time_batch":
                self._repeated_size = num_clusters * batch_size
            else:
                sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                    1, random_state=seed
                )
                if length_batch_uniform:
                    self._repeated_size = (
                        len(sampled_clusters["modeled_seq_len"].unique()) * batch_size
                    )
                else:
                    counts = (
                        sampled_clusters["modeled_seq_len"].value_counts().to_numpy()
                    )
                    # We oversample each length to create a full batch if not divisible by batch_size.
                    epoch_size = np.sum(np.ceil(counts / batch_size) * batch_size)
                    self._repeated_size = int(epoch_size)
            self._log.info(f"Training on {num_clusters} clusters.")
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")
        self._update_sizes()

    def _update_sizes(self):
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def _read_clusters(self):
        pdb_to_cluster = {
            row.pdb_name: row.cluster_ids for row in self._data_csv.itertuples()
        }
        return pdb_to_cluster

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            seed = self.seed + self.epoch
        else:
            seed = self.seed

        if self._sample_mode == "length_batch":
            # Each batch contains multiple proteins of the same length.
            # Each batch contains multiple proteins of the same length.
            # Make 1 batch for each length.
            if self._length_batch_uniform:
                sampled_order = self._data_csv.groupby("modeled_seq_len").sample(
                    self._batch_size, replace=True, random_state=seed
                )
                repeated_indices = sampled_order["index"].tolist()
                # Size may have changed!
                self._repeated_size = len(repeated_indices)
                self._update_sizes()
            # Use all training examples for each length then sample with replacement to complete the batch if necessary.
            else:
                sampled_order = []
                for _, group_elem in self._data_csv.groupby("modeled_seq_len"):
                    group_size = len(group_elem)
                    full_sample = []
                    exhaustive_sample = group_elem.sample(
                        frac=1, random_state=self.epoch
                    )["index"].tolist()
                    full_sample += exhaustive_sample
                    remainder = (
                        self._batch_size - group_size % self._batch_size
                    ) % self._batch_size
                    if remainder > 0:
                        remainder_sample = group_elem.sample(
                            n=remainder, replace=True, random_state=seed
                        )["index"].tolist()
                        full_sample += remainder_sample
                    sampled_order += full_sample
                repeated_indices = sampled_order
                # Size may have changed!
                self._repeated_size = len(repeated_indices)
                self._update_sizes()
        elif self._sample_mode == "time_batch":
            # Each batch contains multiple time steps of the same protein.
            g = torch.Generator()
            g.manual_seed(seed)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()
            repeated_indices = np.repeat(indices, self._batch_size)
        elif self._sample_mode == "cluster_length_batch":
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                1, random_state=self.epoch
            )
            # Sample 1 batch from each length with replacement.
            if self._length_batch_uniform:
                sampled_order = sampled_clusters.groupby("modeled_seq_len").sample(
                    self._batch_size, replace=True, random_state=seed
                )
                repeated_indices = sampled_order["index"].tolist()
                # Size may have changed!
                self._repeated_size = len(repeated_indices)
                self._update_sizes()
            # Use all training examples for each length then sample with replacement to complete the batch if necessary.
            else:
                sampled_order = []
                for _, group_elem in sampled_clusters.groupby("modeled_seq_len"):
                    group_size = len(group_elem)
                    full_sample = []
                    exhaustive_sample = group_elem.sample(frac=1, random_state=seed)[
                        "index"
                    ].tolist()
                    full_sample += exhaustive_sample
                    remainder = (
                        self._batch_size - group_size % self._batch_size
                    ) % self._batch_size
                    if remainder > 0:
                        remainder_sample = group_elem.sample(
                            n=remainder, replace=True, random_state=seed
                        )["index"].tolist()
                        full_sample += remainder_sample
                    sampled_order += full_sample
                repeated_indices = sampled_order
                # Size may have changed!
                self._repeated_size = len(repeated_indices)
                self._update_sizes()
        elif self._sample_mode == "cluster_time_batch":
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby("cluster_ids").sample(
                1, random_state=seed
            )
            dataset_indices = sampled_clusters["index"].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
        else:
            raise ValueError(f"Invalid sample mode: {self._sample_mode}")

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(repeated_indices)
            if padding_size <= len(repeated_indices):
                indices = np.concatenate(
                    (repeated_indices, repeated_indices[:padding_size]), axis=0
                )
            else:
                repeated_indices = np.concatenate(
                    (
                        repeated_indices,
                        np.repeat(
                            repeated_indices,
                            math.ceil(padding_size / len(repeated_indices)),
                        )[:padding_size],
                    ),
                    axis=0,
                )

        else:
            # remove tail of data to make it evenly divisible.
            repeated_indices = repeated_indices[: self.total_size]
        assert len(repeated_indices) == self.total_size

        # subsample
        repeated_indices = repeated_indices[
            self.rank : self.total_size : self.num_replicas
        ]
        self._log.debug(
            f"{self.rank=}: DDP sampler batches first index {repeated_indices[::self._batch_size//self.num_replicas]=}"
        )
        self._log.debug(
            f"{self.rank=}: DDP sampler first batch members {repeated_indices[:self._batch_size//self.num_replicas]=}"
        )

        assert len(repeated_indices) == self.num_samples

        return iter(repeated_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
