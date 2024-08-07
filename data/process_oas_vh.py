# WARNING: This file is not completed and is an aborted attempt at translating PDB files
# from OAS structure data into mmCIF

import argparse
import os
import warnings
from pathlib import Path

from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from exs.sabdab.AbPDB.database_summary import PDBSummary
from exs.sabdab.Database_interface import Database
from joblib import Parallel, delayed


def process_pdb(
    pdb_file: Path,
    pdb_parser: PDBParser,
    rootpath: Path,
):
    try:
        struct = pdb_parser.get_structure(pdb_file.stem, pdb_file)
    except FileNotFoundError:
        warnings.warn(f"PDB file {pdb_file} not found")
        return

    io = MMCIFIO()
    for chain in struct.get_chains():
        if chain.id == "H":
            fname = rootpath / pdb_file.stem[1:3] / f"{pdb_file.stem}_H.cif"
            fname.parent.mkdir(parents=True, exist_ok=True)
            io.set_structure(chain)
            io.save(str(fname))


def process_oas_heavychains(
    pdb_parser: PDBParser,
    oas_data: Path,
    fabsdir: Path,
    n_jobs: int,
):
    oas_structures = oas_data / "structures"

    with Parallel(n_jobs=n_jobs, verbose=5) as parallel:
        results = parallel(
            delayed(process_pdb)(oas_structures / file, pdb_parser, fabsdir)
            for file in os.listdir(oas_structures)
        )


def main(args):
    fabsdir = Path(args.output)
    oas_data = Path(args.oas_data)
    pdb_parser = PDBParser()
    fabsdir.mkdir(parents=True, exist_ok=True)
    process_oas_heavychains(pdb_parser, oas_data, fabsdir, args.n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oas_data", type=str, help="Path to OAS data")
    parser.add_argument(
        "--output",
        type=str,
        default="./mmCIF_oas_vh/",
        help="Output folder for heavy chains",
    )
    parser.add_argument("--n_jobs", type=int, default=2)
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
