from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from exs.pmpnn.dataset_prep.clustering_funcs import (
    cdhit_cluster,
    mmseqs2_easy_cluster,
    read_cdhit_cluster,
    read_mmseqs_cluster,
)
from loguru import logger


def cluster_fabs(
    cdr_seq_df: pd.DataFrame,
    cluster_algo: str,
    cluster_params: list[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Cluster fabs according to cdr overlaps.
    Args:
        cdr_seq_df: Dataframe with cdr sequences for each fab.
        cluster_algo: Clustering algorithm, must be one of "cdhit" or "mmseqs2"
        cluster_params: Parameters for clustering algorithm.
                        If mmseqs2 these parameters are
                        min_seq_id = cluster_params[0],
                        word_length = cluster_params[1]
                        min_seq_length = cluster_params[2]

                        If cdhit these parameters are
                        min_seq_id = cluster_params[0],
                        word_length = cluster_params[1]
                        min_seq_length = cluster_params[2]
        verbose: boolean controlling whether to print summary statistics.

    Returns:
        cdr_seq_df_clustered: Dataframe with cdr sequences for each fab and corresponding cluster ids.
    """
    with TemporaryDirectory() as tempdir:
        output_path = Path(tempdir)
        # Create fasta file for clustering
        fasta_path = output_path / "fab_cdrs.fasta"
        cdr_labels = [f"cdrh{i}" for i in range(1, 4)] + [
            f"cdrl{i}" for i in range(1, 4)
        ]
        cdr_labels = [
            cdr_label for cdr_label in cdr_labels if cdr_label in cdr_seq_df.keys()
        ]
        cdr_seq_df["cdr_concat"] = cdr_seq_df[
            [cdr_label for cdr_label in cdr_labels if cdr_label in cdr_seq_df.keys()]
        ].sum(axis=1)
        with fasta_path.open("w") as file:
            fasta_entries = (
                ">" + cdr_seq_df["pdb_name"] + "\n" + cdr_seq_df["cdr_concat"] + "\n\n"
            )
            for entry in fasta_entries:
                file.write(entry)
        # Perform clustering
        if cluster_algo == "mmseqs2":
            min_similarity = cluster_params[0]
            cov_mode = cluster_params[1]
            coverage = cluster_params[2]

            mmseqs2_easy_cluster(
                fasta_path,
                output_path / "cluster_result",
                cov_mode=cov_mode,
                coverage=coverage,
                min_similarity=min_similarity,
                verbose=True,
            )
            fabid_to_clusterid = read_mmseqs_cluster(
                output_path / "cluster_result_cluster.tsv"
            )
        elif cluster_algo == "cdhit":
            min_seq_id = cluster_params[0]
            word_length = cluster_params[1]
            min_seq_length = cluster_params[2]

            cdhit_cluster(
                fasta_path,
                output_path / "cluster_result",
                min_seq_id=min_seq_id,
                word_length=word_length,
                min_seq_length=min_seq_length,
                verbose=True,
            )
            fabid_to_clusterid = read_cdhit_cluster(
                output_path / "cluster_result.clstr"
            )
        else:
            raise ValueError(
                f"Unrecognised {cluster_algo=}, must be 'cdhit' or 'mmseqs2."
            )
    # Take clusters, create clusterids, and add as a column to cdr_seq_dataframe.
    cdr_seq_df_clustered = cdr_seq_df.copy()
    cdr_seq_df_clustered["cluster_ids"] = cdr_seq_df_clustered["pdb_name"].apply(
        lambda x: fabid_to_clusterid[x]
    )
    if verbose:
        # Print some summary statistics:
        cluster_counts = cdr_seq_df_clustered["cluster_ids"].value_counts()
        logger.info(
            f"number of fabs successfully processed = {len(cdr_seq_df_clustered)}"
        )
        logger.info(f"number of clusters = {len(cluster_counts)}")
        logger.info(f"largest cluster count = {cluster_counts.max()}")
        logger.info(f"smallest cluster count = {cluster_counts.min()}")
        logger.info(f"mean number of cluster members = {cluster_counts.mean()}")
        logger.info(f"median number of cluster members = {cluster_counts.median()}")
        # Counts of unique sequences in largest cluster:
        largest_unique_seq_counts = cdr_seq_df_clustered[
            cdr_seq_df_clustered["cluster_ids"] == cluster_counts.idxmax()
        ]["cdr_concat"].value_counts()
        logger.info(
            "Unique sequence counts in largest cluster:\n"
            f"{largest_unique_seq_counts}"
        )
    return cdr_seq_df_clustered
