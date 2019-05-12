import json
import glob
import h5py
from ase.io import read
from neuralxc.symmetrizer import symmetrizer_factory
from neuralxc.formatter import atomic_shape, system_shape, SpeciesGrouper
from neuralxc.ml.transformer import GroupedPCA, GroupedVarianceThreshold
from neuralxc.ml.transformer import GroupedStandardScaler
from neuralxc.ml import NetworkEstimator as NetworkWrapper
from neuralxc.ml import NXCPipeline
from neuralxc.ml.network import load_pipeline
from neuralxc.preprocessor import Preprocessor
from neuralxc.datastructures.hdf5 import *
from neuralxc.ml.utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from pprint import pprint
from dask.distributed import Client, LocalCluster
from sklearn.externals.joblib import parallel_backend
import time
import os
import shutil
from collections import namedtuple
import hashlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import neuralxc as xc
import sys

def plot_basis(args):
    """ Plots a set of basis functions specified in .json file"""

    basis_instructions = json.loads(open(args.basis,'r').read())
    projector = xc.projector.DensityProjector(np.eye(3),np.ones(3),
        basis_instructions['basis'])

    for spec in basis_instructions['basis']:
        if not len(spec) == 1: continue
        basis = basis_instructions['basis'][spec]
        n = basis_instructions['basis'][spec]['n']
        W = projector.get_W(basis)
        r = np.linspace(0,basis['r_o'],500)
        radials = projector.radials(r,basis,W)
        for rad in radials:
            plt.plot(r, rad)
        plt.show()

def convert_tf(args):
    """ Converts the tensorflow estimator inside a NXCPipeline to a simple
    numpy base estimator"""

    nxc_tf = xc.NeuralXC(args.tf)
    pipeline = nxc_tf._pipeline

    #Needs to do a fake run to build the tensorflow graph
    unitcell = np.eye(3)*20
    grid = [40]*3
    rho = np.zeros(grid)

    species = [key for key in pipeline.get_basis_instructions() if len(key) == 1]
    positions = np.zeros([len(species),3])

    nxc_tf.initialize(unitcell, grid, positions, species)
    nxc_tf.get_V(rho)

    nxc_tf._pipeline.save(args.np, args.override, True)

def add_data_driver(args):
    """ Adds data to hdf5 file"""
    try:
        file = h5py.File(args.hdf5 ,'r+')
    except OSError:
        file = h5py.File(args.hdf5 ,'w')

    i,j,k = [(None if a == '' else int(a)) for a in args.slice.split(':')] +\
        [None]*(3-len(args.slice.split(':')))

    ijk = slice(i,j,k)

    def obs(which):
        if which == 'energy':
            if args.traj:
                add_species(file, args.system, args.traj)
                energies = np.array([a.get_potential_energy()\
                 for a in read(args.traj,':')])[ijk]
                add_energy(file, energies, args.system, args.method, args.override)
            else:
                raise Exception('Must provide a trajectory file')
                file.close()
        elif which == 'forces':
            if args.traj:
                add_species(file, args.system, args.traj)
                forces = np.array([a.get_forces()\
                 for a in read(args.traj,':')])[ijk]
                add_forces(file, forces, args.system, args.method, args.override)
            else:
                raise Exception('Must provide a trajectory file')
                file.close()
        elif which == 'density':
            add_species(file, args.system, args.traj)
            species = file[args.system].attrs['species']
            data = np.load(args.density)[ijk]
            add_density((args.density.split('/')[-1]).split('.')[0], file, data,
                args.system, args.method, args.override)
        else:
            raise Exception('Option {} not recognized'.format(which))

    if args.density and not 'density' in args.add:
        args.add.append('density')
    for observable in args.add:
        obs(observable)

    file.close()


def split_data_driver(args):
    """ Split dataset (or all data inside a group) by providing slices"""
    file = h5py.File(args.hdf5 ,'r+')

    i,j,k = [(None if a == '' else int(a)) for a in args.slice.split(':')] +\
        [None]*(3-len(args.slice.split(':')))

    ijk = slice(i,j,k)


    root = args.group
    if not root[0] == '/': root = '/' + root

    def collect_all_sets(file, path):
        sets = {}
        if isinstance(file[path],h5py._hl.dataset.Dataset):
            return {path: file[path]}
        else:
            for key in file[path]:
                sets.update(collect_all_sets(file, path + '/' + key))
        return sets

    all_sets = collect_all_sets(file, root)
    split_sets = {}
    comp_sets = {}
    length = -1
    for path in all_sets:
        new_len = len(all_sets[path][:])
        if length == -1:
            length = new_len
        elif new_len != length:
            raise Exception('Datasets contained in group dont have consistent lengths')
        idx = path.find(args.group) + len(args.group)
        new_path = path[:idx] +'/' + args.label + path[idx:]
        if args.comp != '':
            idx = path.find(args.group) + len(args.group)
            comp_path = path[:idx] +'/' + args.comp + path[idx:]
            comp_sets[comp_path] = all_sets[path][:].tolist()
            del comp_sets[comp_path][ijk]
        split_sets[new_path] = all_sets[path][ijk]

    for new_path in split_sets:
        file.create_dataset(new_path, data = split_sets[new_path])

    for new_path, path in zip(split_sets, all_sets):
        file['/'.join(new_path.split('/')[:-1])].attrs.update(file['/'.join(path.split('/')[:-1])].attrs)
    if comp_sets:
        for comp_path in comp_sets:
            file.create_dataset(comp_path, data = comp_sets[comp_path])
        for new_path, path in zip(comp_sets, all_sets):
                file['/'.join(new_path.split('/')[:-1])].attrs.update(file['/'.join(path.split('/')[:-1])].attrs)
    # print(split_sets)

def delete_data_driver(args):
    """ Deletes data in hdf5 file"""
    file = h5py.File(args.hdf5 ,'r+')
    root = args.group
    if not root[0] == '/': root = '/' + root
    del file[root]

def parse_sets_input(path):
    """ Reads a file containing the sets used for fitting

    Parameters
    ----------
    path: str
        Path to file containing dataset names

    Returns
    --------
    hdf5, list
        hdf5[0] : datafile location
        hdf5[1],hdf5[2]: lists of baseline(,target) datasets
    """
    hdf5=['',[],[]]
    with open(path, 'r') as setsfile:
        line = setsfile.readline().rstrip()
        hdf5[0] = line #datafile location
        line = setsfile.readline().rstrip()
        while(line):
            split = line.split()
            hdf5[1].append(split[0])
            hdf5[2].append(split[1])
            line = setsfile.readline().rstrip()
    return hdf5

def fit_driver(args):
    """ Fits a NXCPipeline to the provided data
    """
    inputfile = args.config
    preprocessor = args.preprocessor

    if args.sets != '':
        hdf5 = parse_sets_input(args.sets)
    else:
        hdf5 = args.hdf5

    mask = args.mask

    if not mask:
        inp = json.loads(open(inputfile,'r').read())
        pre = json.loads(open(preprocessor,'r').read())
    else:
        inp = {}
        pre = {}

    best_model = get_grid_cv(hdf5, preprocessor, inputfile, mask)
    if not mask:
        start = time.time()
        try:
            os.mkdir('.tmp')
        except FileExistsError:
            pass

        datafile = h5py.File(hdf5[0],'r')
        basis_key = basis_to_hash(pre['basis'])
        data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, args.cutoff)
        if args.sample != '':
            sample = np.load(args.sample)
            data = data[sample]
            print("Using sample of size {}".format(len(sample)))
        cluster = LocalCluster(n_workers = inp.get('n_workers',1),
        threads_per_worker=inp.get('threads_per_worker',1))
        # cluster = LocalCluster(processes=False)
        client = Client(cluster)
        if inp.get('n_workers',1)==1 and inp.get('threads_per_worker',1)==1:
            backend = 'loky'
        else:
            backend = 'dask'
        print("BACKEND: ", backend)
        with parallel_backend(backend):
            print('======Hyperparameter search======')

            best_model.fit(data)
        # best_model.fit(list(range(len(atoms))))
        end = time.time()
        print('Took {}s'.format(end-start))
        open('best_params.json','w').write(json.dumps(best_model.best_params_, indent=4))
        pd.DataFrame(best_model.cv_results_).to_csv('cv_results.csv')
        best_params_ = best_model.best_params_
        best_estimator = best_model.best_estimator_.steps[0][1].start_at(2)
        best_estimator.basis_instructions =  pre['basis']
        best_estimator.symmetrize_instructions = {'symmetrizer_type':'casimir'}
        best_estimator.save('best_model',True)

def sample_driver(args):
    """ Given a dataset, perform sampling in feature space"""

    preprocessor = args.preprocessor
    hdf5 = args.hdf5

    pre = json.loads(open(preprocessor,'r').read())

    datafile = h5py.File(hdf5[0],'r')
    basis_key = basis_to_hash(pre['basis'])
    data = load_sets(datafile, hdf5[1], hdf5[1], basis_key, args.cutoff)
    basis = pre['basis']
    symmetrizer_instructions = {'symmetrizer_type' :'casimir'}
    symmetrizer_instructions.update({'basis' : basis})
    species =  [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]
    spec_group = SpeciesGrouper(basis, species)
    symmetrizer = symmetrizer_factory(symmetrizer_instructions)

    sampler_pipeline = Pipeline([('spec_group', spec_group),('symmetrizer',symmetrizer),
        ('sampler', SampleSelector(args.size))])
    sample = sampler_pipeline.predict(data)
    np.save(args.dest, np.array(sample))

def eval_driver(args):
    """ Evaluate fitted NXCPipeline on dataset and report statistics
    """
    preprocessor = args.preprocessor
    hdf5 = args.hdf5

    pre = json.loads(open(preprocessor,'r').read())

    datafile = h5py.File(hdf5[0],'r')
    basis_key = basis_to_hash(pre['basis'])
    data = load_sets(datafile, hdf5[1], hdf5[2], basis_key, args.cutoff)

    if not args.model == '':
        model = xc.NeuralXC(args.model)._pipeline
        basis = model.get_basis_instructions()
        symmetrizer_instructions = model.get_symmetrize_instructions()
        symmetrizer_instructions.update({'basis' : basis})
        species =  [''.join(find_attr_in_tree(datafile, hdf5[1], 'species'))]
        spec_group = SpeciesGrouper(basis, species)
        symmetrizer = symmetrizer_factory(symmetrizer_instructions)
        print('Symmetrizer instructions', symmetrizer_instructions)
        pipeline = NXCPipeline([('spec_group', spec_group), ('symmetrizer', symmetrizer)
            ] + model.steps, basis_instructions=basis,
             symmetrize_instructions=symmetrizer_instructions)

        targets = data[:,-1].real
        predictions = pipeline.predict(data)[0]
        print(predictions.shape)
        print(targets.shape)
        dev = (predictions - targets)
    else:
        dev = data[:,-1].real
    dev0 = np.abs(dev - np.mean(dev))
    results = {'mean deviation' : np.mean(dev).round(4), 'rmse': np.std(dev).round(4),
               'mae' : np.mean(dev0).round(4),'max': np.max(dev0).round(4)}

    pprint(results)
    if args.plot:
        plt.plot(targets.flatten(),predictions.flatten(),ls ='',marker='.')
        plt.show()

def pre_driver(args):
    """ Preprocess electron densities obtained from electronic structure
    calculations
    """
    preprocessor = args.preprocessor
    dest = args.dest
    xyz = args.xyz
    mask = args.mask

    if not mask:
        pre = json.loads(open(preprocessor,'r').read())
    else:
        pre = {}

    if 'traj_path' in pre and pre['traj_path'] != '':
        atoms = read(pre['traj_path'], ':')
        trajectory_path = pre['traj_path']
    elif xyz != '':
        atoms = read(xyz, ':')
        trajectory_path = xyz
    else:
        raise ValueError('Must specify path to to xyz file')

    preprocessor = get_preprocessor(preprocessor, mask, xyz)
    if not mask:
        start = time.time()

        if 'hdf5' in dest:
            dest_split = dest.split('/')
            file, system, method = dest_split + ['']*(3-len(dest_split))
            workdir = '.tmp'
            delete_workdir = True
        else:
            workdir = dest
            delete_workdir = False

        try:
            os.mkdir(workdir)
        except FileExistsError:
            delete_workdir = False
            pass
        print('======Projecting onto basis sets======')
        basis_grid = get_basis_grid(pre)['preprocessor__basis_instructions']

        for basis_instr in basis_grid:
            preprocessor.basis_instructions = basis_instr
            filename = os.path.join(workdir,basis_to_hash(basis_instr) + '.npy')
            data = preprocessor.fit_transform(None)
            np.save(filename, data)
            if 'hdf5' in dest:
                data_args = namedtuple(\
                'data_ns','hdf5 system method density slice add traj override')(\
                file,system,method,filename, ':',[],trajectory_path, True)
                add_data_driver(data_args)


        if delete_workdir:
            shutil.rmtree(workdir)