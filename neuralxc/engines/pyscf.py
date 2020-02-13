import pyscf
from pyscf import gto, dft
from pyscf.dft.libxc import define_xc_
from pyscf.dft import RKS
from pyscf.scf.chkfile import load_scf
from pyscf.lib.numpy_helper import NPArrayWithTag
import numpy as np
from ase.io import read, write
import argparse
import os
from ase import Atoms
import neuralxc


def compute_KS(atoms, path='pyscf.chkpt', basis='ccpvdz', xc='PBE', nxc=''):
    """ Run Kohn-Sham (KS) DFT calculation for system specified in atoms.

    Parameters
    ----------
    atoms: ase.Atoms
        ase.Atoms object describing the system
    path: str
        path to save PySCF checkpoint to (default: pyscf.chkpt)
    basis: str
        basis set, use PySCF nomenclature (default: ccpvdz)
    xc: str
        xc-functional, use PySCF nomenclature (default: PBE)
    nxc: str
        path to neuralxc-functional (default '': do not use NXC)
    """

    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]

    mol = gto.M(atom=mol_input, basis=basis)
    mf = dft.RKS(mol)
    mf.set(chkfile=path)
    mf.xc = xc
    if not nxc is '':
        model = neuralxc.get_nxc_adapter('pyscf', nxc)
        model.initialize(mol)
        mf.get_veff = _veff_mod(mf, model)
    mf.kernel()
    return mf, mol


def _veff_mod(mf, model):
    """
    Modified effective potential that includes NeuralXC functional
    """
    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        veff = pyscf.dft.rks.get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        vnxc = NPArrayWithTag(veff.shape)
        nxc = model.get_V(dm)
        vnxc[:, :] = nxc[1][:, :]
        vnxc.exc = nxc[0]
        vnxc.ecoul = 0
        veff[:, :] += vnxc[:, :]
        veff.exc += vnxc.exc
        return veff

    return get_veff
