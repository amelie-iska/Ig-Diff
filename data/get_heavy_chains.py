# WARNING: This file is not completed and is an aborted attempt at translating SAbDab fabs into mmCIF

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB.mmcifio import MMCIFIO
from exs.sabdab.AbPDB.database_summary import PDBSummary
from exs.sabdab.Database_interface import Database
from joblib import Parallel, delayed


def get_database(local_db: str, update_from_s3: bool):

    # test if local_db exists and has db_summary, if not or if local_db="", download it with the code below

    if local_db == "":
        sabdab_db = Database(update_from_s3=update_from_s3)
    else:
        local_summaries_dirpath = Path(local_db) / "Summaries"
        local_summaries_dirpath.mkdir(parents=True, exist_ok=True)
        local_summaries_filepath = local_summaries_dirpath / "db_summary.dat"
        if update_from_s3 or not local_summaries_filepath.exists():

            print("Downloading summaries file from s3.")
            s3path = S3Path(
                "s3://exs-biologicsteam-data/sabdab/database/Summaries/db_summary.dat"
            )
            s3path.download_to(local_summaries_filepath)

        sabdab_db = Database(path=local_db, update_from_s3=update_from_s3)
    return sabdab_db


def process_pdb(
    pdb_summary: PDBSummary,
    sabdab_db: Database,
    rootpath: Path,
    res_cutoff: Optional[float] = None,
):
    try:
        pdb_details = sabdab_db.fetch(pdb_summary.pdb)
    except FileNotFoundError:
        warnings.warn(f"PDB file {pdb_summary.pdb} not found")
        return
    if not pdb_details.get_resolution() or np.isnan(pdb_details.get_resolution()):
        rejections = ["no resolution"]
        return
    if res_cutoff:
        if pdb_details.get_resolution() > res_cutoff:
            rejections = ["rescutoff condition"]
            return
    io = MMCIFIO()

    for fab_detail in pdb_details.get_fabs():

        try:
            ab_struct = fab_detail.get_structure(scheme="imgt")
            vh = ab_struct.get_VH()
        except Exception:
            print(pdb_details.identifier)
            print(fab_detail)
            continue
        fname = rootpath / pdb_summary.pdb[1:3] / f"{pdb_summary.pdb}_{vh.id}.cif"
        fname.parent.mkdir(parents=True, exist_ok=True)
        io.set_structure(vh)
        io.save(str(fname))


def process_sabdab_heavychains(
    sabdab_db: Database,
    fabsdir: Path,
    n_jobs: int,
):

    summary_dict = sabdab_db.get_summary()
    pdb_summaries = list(summary_dict.values())

    with Parallel(n_jobs=n_jobs, verbose=5) as parallel:
        results = parallel(
            delayed(process_pdb)(pdb_summary, sabdab_db, fabsdir)
            for pdb_summary in pdb_summaries
        )


def main(args):
    fabsdir = Path(args.output)
    sabdab_db = get_database(args.sabdab, args.update_from_s3)
    fabsdir.mkdir(parents=True, exist_ok=True)
    process_sabdab_heavychains(sabdab_db, fabsdir, args.n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sabdab", type=str, default="", help="Path to SAbDab database"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mmCIF_vh/",
        help="Output folder for heavy chains",
    )
    # parser.add_argument('--res_cutoff', type=float, default=None, help='Optional maximum resolution cut off to discard pdb files.')
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--update_from_s3", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
#
# # code from dan
# from exs.sabdab.Database_interface import Database
# from exs.sabdab.AbPDB.Select import fv_only
#
# db = Database(local_dbpath='/Users/dcutting/Data/sabdab_db')
# pdb = db.fetch('7chh')
# ab = pdb['JK']
# struct = ab.get_structure(scheme='imgt',definition='north')
# H_chain = struct.get_VH()
# H_chain.save('./my_hchain.pdb',selection=fv_only())
#
#
# #things in my historys
# from Bio.PDB.mmcifio import MMCIFIO
# io=MMCIFIO()
# io.set_structure(s)
# io.save('test.cif')
# for fab_detail in pdb.get_fabs():
#     vh = fab_detail.get_structure(scheme='imgt').get_VH()
#     io.set_structure(vh)
#     vh.save(fname, selection=fv_only())
#     print(vhs[-1])
# io.set_structure(vhs[0].get_VH())
# io.save('test_h.cif')
# vhs[0].get_VH().save('test_h.pdb',selection=fv_only())
# io.set_structure(vhs[0].get_VH())
