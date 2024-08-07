"""Library for parsing different data structures."""
import numpy as np
from Bio.PDB.Chain import Chain

from data import protein, residue_constants

Protein = protein.Protein


def process_chain(chain: Chain, chain_idx: int) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    # TODO: add code to account for missing residues when creating residue index.
    for idx, res in enumerate(chain):
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(idx)
        b_factors.append(res_b_factors)
    chain_idxs = np.ones(len(residue_index), dtype=np.int32) * chain_idx

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_idxs,
        b_factors=np.array(b_factors),
    )
