"""Script for preprocessing PDB files.

WARNING: NOT TESTED WITH SE(3) DIFFUSION.
This is example code of how to preprocess PDB files.
It does not process extra features that are used in process_pdb_dataset.py.
One can use the logic here to create a version of process_pdb_dataset.py
that works on PDB files.

"""

import argparse
import dataclasses
import functools as fn
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Optional

import mdtraj as md
import numpy as np
import pandas as pd
import tqdm
from Bio.SeqUtils import seq1
from exs.sabdab import Antibody
from joblib import Parallel, delayed
from loguru import logger

from data import errors, parsers
from data import utils as du


def process_file(
    file_path: Path, write_dir: Path, chains_to_process: str = "HL"
) -> list[dict[str, Any]]:
    """Processes pdb files containing antibodies into pickle files of features and create metadata entries.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.
        chains_to_process: String specifying whether to process heavy chains ("H"), light chains ("L") or both ("HL")

    Returns:
        metadata_list: list of metadata entries corresponding to each fab in the pdb file.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    if chains_to_process.upper() not in ["H", "L", "HL", "LH"]:
        raise ValueError(
            f'{chains_to_process=} but must be one of "H", "L", "HL" or "LH".'
        )

    pdb_name = file_path.name.replace(".pdb", "")
    ab = Antibody.from_structure(file_path)
    structure = ab.structure
    metadata_list = []
    # Process each fab in pdb file as a separate entry
    for fv in structure.get_fvs():
        metadata = {}
        metadata["pdb_name"] = f"{pdb_name}_{fv.get_id()}"

        processed_path = write_dir / "processed_pkls" / f"{pdb_name}_{fv.get_id()}.pkl"
        processed_path.parent.mkdir(exist_ok=True, parents=True)
        metadata["processed_path"] = processed_path.relative_to(write_dir)
        metadata["raw_path"] = str(file_path)

        fv_chain_to_process = []
        if "H" in chains_to_process.upper():
            heavy_chain = fv.heavy
            if heavy_chain is None:
                logger.warning(
                    f"No heavy chain found for {pdb_name}_{fv.get_id()}, skipping."
                )
                continue
            fv_chain_to_process.append(heavy_chain)

            for cdr in [f"cdrh{i}" for i in range(1, 4)]:
                try:
                    region_sequence = heavy_chain.get_region_sequence(
                        cdr, definition="north"  # type:ignore
                    )
                    metadata[cdr] = region_sequence
                except LookupError as e:
                    logger.warning(
                        f"Not all heavy cdrs present for {pdb_name}_{fv.get_id()}, skipping."
                    )
                    continue
        if "L" in chains_to_process.upper():
            light_chain = fv.light
            if light_chain is None:
                logger.warning(
                    f"No light chain found for {pdb_name}_{fv.get_id()}, skipping."
                )
                continue
            fv_chain_to_process.append(light_chain)

            for cdr in [f"cdrl{i}" for i in range(1, 4)]:
                try:
                    region_sequence = light_chain.get_region_sequence(
                        cdr, definition="north"  # type:ignore
                    )
                    metadata[cdr] = region_sequence
                except LookupError as e:
                    logger.warning(
                        f"Not all light cdrs present for {pdb_name}_{fv.get_id()}, skipping."
                    )
                    continue

        # Extract features
        struct_feats = []
        all_seqs = set()
        num_chains = 0

        for chain_idx, chain in enumerate(fv_chain_to_process):
            num_chains += 1
            chain_prot = parsers.process_chain(chain, chain_idx)
            chain_dict = dataclasses.asdict(chain_prot)
            all_seqs.add(tuple(chain_dict["aatype"]))
            struct_feats.append(chain_dict)
        metadata["num_chains"] = num_chains
        if len(all_seqs) == 1:
            metadata["quaternary_category"] = "homomer"
        else:
            metadata["quaternary_category"] = "heteromer"
        complex_feats = du.concat_np_features(struct_feats, False)
        # Do this outside the loop otherwise chains and centered
        # independently which we don't want.
        complex_feats = du.center_and_scale_coords(complex_feats)

        metadata["oligomeric_count"] = num_chains  # oligomeric_count
        if num_chains == 1:
            metadata[
                "oligomeric_detail"
            ] = "monomeric"  # oligomeric_detail #type:ignore
        else:
            metadata["oligomeric_detail"] = None  # oligomeric_detail

        # Process geometry features
        complex_aatype = complex_feats["aatype"]
        metadata["seq_len"] = len(complex_aatype)

        # adapting from process_pdb_dataset.py
        modeled_idx = np.where(complex_aatype != 20)[0]
        if np.sum(complex_aatype != 20) == 0:
            logger.error("No modeled residues")
            raise errors.LengthError("No modeled residues")
        min_modeled_idx = np.min(modeled_idx)
        max_modeled_idx = np.max(modeled_idx)
        metadata["modeled_seq_len"] = max_modeled_idx - min_modeled_idx + 1
        complex_feats["modeled_idx"] = modeled_idx
        # if complex_aatype.shape[0] > max_len:
        #     print(f"Too long {complex_aatype.shape[0]}")
        #     raise errors.LengthError(f"Too long {complex_aatype.shape[0]}")
        try:
            # MDtraj
            # Have to resave just the fab as we want metrics on that alone.
            with tempfile.TemporaryDirectory() as tempdir:
                temppath = Path(tempdir)
                temp_fabpath = str(temppath / f"{pdb_name}_{fv.get_id()}.pdb")
                if "H" in chains_to_process:
                    if "L" in chains_to_process:
                        fv.save(temp_fabpath)
                    else:
                        fv.heavy.save(temp_fabpath)
                else:
                    fv.light.save(temp_fabpath)
                traj = md.load(temppath / f"{pdb_name}_{fv.get_id()}.pdb")
                # SS calculation
                pdb_ss = md.compute_dssp(traj, simplified=True)
                # DG calculation
                pdb_dg = md.compute_rg(traj)

        except Exception as e:
            logger.error(f"Mdtraj failed with error {e}")
            raise ValueError("Mdtraj error")

        metadata["coil_percent"] = np.sum(pdb_ss == "C") / metadata["modeled_seq_len"]
        metadata["helix_percent"] = np.sum(pdb_ss == "H") / metadata["modeled_seq_len"]
        metadata["strand_percent"] = np.sum(pdb_ss == "E") / metadata["modeled_seq_len"]

        # Radius of gyration
        metadata["radius_gyration"] = pdb_dg[0]
        # Write features to pickles.
        du.write_pkl(str(processed_path), complex_feats)

        # Add metadata to list
        metadata_list.append(metadata)
    return metadata_list


def process_serially(
    all_paths: list[Path], write_dir: Path, chains_to_process: str = "HL"
):
    all_metadata = []
    for i, file_path in enumerate(tqdm.tqdm(all_paths)):
        try:
            start_time = time.time()
            metadata_list = process_file(
                file_path, write_dir, chains_to_process=chains_to_process
            )
            elapsed_time = time.time() - start_time
            logger.debug(f"Finished {file_path} in {elapsed_time:2.2f}s")
            all_metadata.extend(metadata_list)
        except errors.DataError as e:
            logger.debug(f"Failed {file_path}: {e}")
    return all_metadata


def process_fn(
    file_path: Path, write_dir: Path, chains_to_process: str = "HL"
) -> Optional[list[dict[str, Any]]]:
    try:
        start_time = time.time()
        metadata_list = process_file(
            file_path, write_dir, chains_to_process=chains_to_process
        )
        elapsed_time = time.time() - start_time
        logger.debug(f"Finished {file_path} in {elapsed_time:2.2f}s")
        return metadata_list
    except errors.DataError as e:
        logger.debug(f"Failed {file_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error {e} from {file_path=}.")
        return None


def processing_main(
    pdb_dir: Path,
    write_dir: Path,
    chains_to_process: str,
    num_processes: int,
    debug: bool,
) -> None:
    all_file_paths = list(pdb_dir.glob("*.pdb"))
    total_num_paths = len(all_file_paths)
    write_dir.mkdir(exist_ok=True, parents=True)
    if debug:
        metadata_file_name = "metadata_debug.csv"
    else:
        metadata_file_name = "metadata.csv"
    metadata_path = write_dir / metadata_file_name
    logger.info(f"Files will be written to {write_dir}")

    # Process each pdb file
    if num_processes == 1 or debug:
        all_metadata = process_serially(
            all_file_paths, write_dir, chains_to_process=chains_to_process
        )
    else:
        _process_fn = fn.partial(
            process_fn, chains_to_process=chains_to_process, write_dir=write_dir
        )
        with Parallel(n_jobs=num_processes, verbose=5) as parallel:
            all_metadata = parallel(
                delayed(process_fn)(
                    file_path=filepath,
                    write_dir=write_dir,
                    chains_to_process=chains_to_process,
                )
                for filepath in all_file_paths
            )
        failed_files = all_metadata.count(None)
        logger.info(f"Number of failed files is {failed_files}")
        all_metadata = [y for x in all_metadata if x is not None for y in x]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    logger.info(
        f"Finished processing {succeeded} structures from {total_num_paths} files"
    )


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Define the parser
    parser = argparse.ArgumentParser(description="PDB processing script.")
    parser.add_argument("--pdb_dir", help="Path to directory with PDB files.", type=str)
    parser.add_argument(
        "--num_processes", help="Number of processes.", type=int, default=50
    )
    parser.add_argument(
        "--write_dir",
        help="Path to write results to.",
        type=str,
        default="./preprocessed_pdbs",
    )
    parser.add_argument("--debug", help="Turn on for debugging.", action="store_true")
    parser.add_argument(
        "--verbose", help="Whether to log everything.", action="store_true"
    )
    parser.add_argument(
        "--chains_to_process",
        type=str,
        default="HL",
        help="Specifies whether to process heavy chain 'H', light chain 'L' or both 'HL'",
    )

    args = parser.parse_args()
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    processing_main(
        pdb_dir=Path(args.pdb_dir),
        write_dir=Path(args.write_dir),
        chains_to_process=args.chains_to_process,
        num_processes=args.num_processes,
        debug=args.debug,
    )
