"""Utility functions for experiments."""
import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

from openfold.utils import rigid_utils

Rigid = rigid_utils.Rigid


def get_ddp_info() -> dict[str, int]:
    """
    Assumes that DDP has been launched with torchrun
    Returns:
        Dictionary containing information about the process running in DDP.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    node_id = rank // local_world_size
    return {
        "node_id": node_id,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "rank": rank,
        "world_size": world_size,
    }


def flatten_dict(raw_dict: dict) -> list[tuple]:
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([(f"{k}:{i}", j) for i, j in flatten_dict(v)])
        else:
            flattened.append((k, v))
    return flattened


def t_stratified_loss(
    batch_t: torch.Tensor,
    batch_loss: torch.Tensor,
    num_bins: int = 5,
    loss_name: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Stratify loss by binning t. Returns two dictionaries containing the total loss for examples in the bin, as well as the bin counts.
    The bin counts can be used when if we need to compute the average loss across multiple processes.
    Args:
        batch_t: Tensor containing the time that the example was diffused to for the batch.
        batch_loss: Tensor containing the loss that was obtained for the sample.
        num_bins: How many bins for time. Default 5.
        loss_name: Optional name for the loss, this will be prepended to the dictionary key.
    Returns:
        stratified_loss: Dictionary mapping bin range to bin total loss.
        stratified_counts: Dictionary mapping bin range to bin count.
    """
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses, minlength=num_bins)
    t_binned_n = np.bincount(bin_idx, minlength=num_bins)
    stratified_losses = {}
    stratified_counts = {}
    if loss_name is None:
        loss_name = "loss"
    for i in range(num_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        stratified_losses[t_range] = t_binned_loss[i]
        stratified_counts[t_range] = t_binned_n[i]
    return stratified_losses, stratified_counts


def get_sampled_mask(
    contigs: str,
    length: Optional[list[int]],
    rng: Optional[np.random.Generator] = None,
    num_tries: int = 1000000,
) -> tuple[list[str], int, int]:
    """
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.

    Args:
        contigs:
        length:
        rng:
        num_tries:
    """
    length_compatible = False
    count = 0
    inpaint_chains = 0
    sampled_mask = []
    sampled_mask_length = 0
    while length_compatible is False:
        inpaint_chains = 0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        # allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f"{contig_list[-1]},0"
        for con in contig_list:
            if (
                all([i[0].isalpha() for i in con.split(",")[:-1]])
                and con.split(",")[-1] == "0"
            ):
                # receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                # chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if "-" in subcon:
                            sampled_mask_length += (
                                int(subcon.split("-")[1])
                                - int(subcon.split("-")[0][1:])
                                + 1
                            )
                        else:
                            sampled_mask_length += 1

                    else:
                        if "-" in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(
                                    int(subcon.split("-")[0]), int(subcon.split("-")[1])
                                )
                            else:
                                length_inpaint = random.randint(
                                    int(subcon.split("-")[0]), int(subcon.split("-")[1])
                                )
                            subcon_out.append(f"{length_inpaint}-{length_inpaint}")
                            sampled_mask_length += length_inpaint
                        elif subcon == "0":
                            subcon_out.append("0")
                        else:
                            length_inpaint = int(subcon)
                            subcon_out.append(f"{length_inpaint}-{length_inpaint}")
                            sampled_mask_length += int(subcon)
                sampled_mask.append(",".join(subcon_out))
        # check length is compatible
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count += 1
        if count == num_tries:  # contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains
