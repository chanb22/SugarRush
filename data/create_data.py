# SugarRush Data Generation Program
# Channing Bellamy (BLLCHA013)
# University of Cape Town
# Based on GeqShift (arXiv:2311.12657, University of Stockholm, Sweden)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
import pickle
import numpy as np
import torch
import os 
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
from CarbonDataset import Carbons13C
from HydrogenDataset import Hydrogens1H

from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader

import sys

def get_cut_off_graph(edge_index, edge_attr, p, cut_off = 6.0):
    row, col = edge_index
    dist = torch.sqrt(torch.sum((p[row]- p[col])**2, dim = 1))
    mask = dist <= cut_off
    edge_index = edge_index[:,mask]
    edge_attr = edge_attr[mask]
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce  = "max")
    return edge_index, edge_attr

def generate_conformations(m_, nbr_confs = 10):
    params = Chem.rdDistGeom.ETKDGv3()
#    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.01
    params.numThreads = 0
    params.enforceChirality = True
    params.maxIterations = 10000

    params.randomSeed = 42

    Chem.SanitizeMol(m_)
    m_ = Chem.AddHs(m_, addCoords=True)

    print(f"Embedding...")
    em = Chem.rdDistGeom.EmbedMultipleConfs(m_, numConfs=nbr_confs*2, params=params)
    ps = AllChem.MMFFGetMoleculeProperties(m_, mmffVariant='MMFF94')
    
    energies = []
    for conf in m_.GetConformers():
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m_, ps, confId=conf.GetId())
        
        if isinstance(ff, type(None)):
            print(f"ERROR: ff isinstance None!")
            continue
        energy = ff.CalcEnergy()
        energies.append(energy)

    m_ = Chem.RemoveHs(m_)
    if em == -1:
            print(f"ERROR: em == -1")
            conformations = []            
            for i, c in enumerate(m_.GetConformers()):
                xyz = c.GetPositions()
                conformations.append(xyz)
                if i >9:
                    return conformations
    energies = np.array(energies)
    ind = energies.argsort()[:nbr_confs]
    energies = energies[ind]
    conformations = []
    for i, c in enumerate(m_.GetConformers()):

        if i not in ind:
            continue
        xyz = c.GetPositions()
        conformations.append(xyz)

    return conformations, energies, m_


def create_conformer(mols, nbr_confs = 10):
    for mol in mols:
        name = mol.GetProp("_Name")

        name = name.replace("->", "")
        name = name.replace("(","")
        name = name.replace(")","")

        path = "conformations//" + name + ".pickle"
        if os.path.isfile(path):
            continue
        print(f"({mols.index(mol)}/{len(mols)}) Generating conformers for {name}")
        conformations, _,_ = generate_conformations(mol,nbr_confs)
        if len(conformations) < nbr_confs:
            print(f"WARNING: only {len(conformations)}/{nbr_confs} conformations for {name}, skipping...")
            continue
        with open(path, 'wb') as handle:
            pickle.dump(conformations, handle)

def train_test_split(mols, fold):
    N_t = len(mols)
    indices = range(N_t)
    indices_kfold = np.array_split(indices, 10)[fold]
    train_mask = np.ones(N_t, dtype=bool)
    train_mask[indices_kfold] = False
    test_mask = [not(a) for a in train_mask]
    train_datas = [data for data, m in zip(mols, train_mask) if m]
    test_datas = [data for data, m in zip(mols, test_mask) if m]   
#    return mols, mols
    return  train_datas, test_datas

def add_conformations_to_train_dataset(dataset):
    new_dataset = []
    for d_ in dataset:
        name = d_.name
        name = name.replace("->","")
        name = name.replace(")","")
        name = name.replace("(","")
        conf_path = "conformations//" + name + ".pickle"
        if not os.path.exists(conf_path):
            print(f"WARNING: train dataset - no conformer for {name} found, skipping...")
            continue

        with open(conf_path, 'rb') as handle:
            distencies = pickle.load(handle)
            for dis_ in distencies:
                new_data = d_.clone()
                new_data.pos = torch.tensor(dis_)
                edge_index, edge_attr = get_cut_off_graph(new_data.edge_index, new_data.edge_attr, new_data.pos ,cut_off=6.)
                new_data.edge_index = edge_index
                new_data.edge_attr = edge_attr
                new_dataset.append(new_data)
    new_dataset_ = InMemoryDataset()
    data, slices, _ = collate(
                new_dataset[0].__class__,
                data_list=new_dataset,
                increment=False,
                add_batch=False,
            )
    new_dataset_.data = data

    new_dataset_.slices = slices
    return new_dataset_

def add_conformations_to_test_dataset(dataset):
    test_datas = []
    for data in dataset:
        datas_ = []
        name = data.name
        name = name.replace("->","")
        name = name.replace("(","")
        name = name.replace(")","")

        conf_path = "conformations//" + name + ".pickle"
        k = 0
        if os.path.exists(conf_path):
                with open(conf_path, 'rb') as handle:
                        distencies = pickle.load(handle)
                for j, dis_ in enumerate(distencies):
                        d = data.clone()
                        
                        pos = torch.from_numpy(dis_).float()
                        d.pos = pos
                        edge_index, edge_attr = get_cut_off_graph(d.edge_index, d.edge_attr, d.pos ,cut_off=6.)
                        d.edge_index = edge_index
                        d.edge_attr = edge_attr
                        datas_.append(d)
                        k = k + 1
                if len(datas_) == 0:
                    print(f"WARNING: test dataset - no conformers for {name}, skipping...")
                    continue

        else:
                print(f"WARNING: test dataset - no conformer for {name} found, skipping...")
                continue
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)
        test_datas.append( {"name":data.name, "n_nmr":torch.sum(mask).item(), "n_mols": k,
                                        "loader" : DataLoader(datas_, batch_size = len(datas_), shuffle=False )})
    return test_datas

def main(path_to_data: str, num_confs: int):
    np.random.seed(42)
    torch.manual_seed(42)

    mols_mono_casper = []
    mols_di_casper = []
    mols_tri_casper = []

    mols_mono_glyconmr = []
    mols_di_glyconmr = []
    mols_tri_glyconmr = []
    mols_tetra_glyconmr = []
    mols_oligo_glyconmr = []

    mols_mono_csdb = []
    mols_di_csdb = []
    mols_tri_csdb = []
    mols_tetra_csdb = []
    mols_oligo_csdb = []
    mols_poly_csdb = []

    with Chem.SDMolSupplier(path_to_data) as suppl:
        for mol in suppl:
            source = mol.GetProp("Source")
            saccharide_class = mol.GetProp("Class")
            if source == "CASPER":
                if saccharide_class == "mono":
                    mols_mono_casper.append(mol)
                elif saccharide_class == "di":
                    mols_di_casper.append(mol)
                elif saccharide_class == "tri":
                    mols_tri_casper.append(mol)
                else:
                    print(f"ERROR: unknown mol saccharide class for CASPER source - not in mono, di, or tri. Aborting!")
                    sys.exit()
            elif source == "GlycoNMR":
                if saccharide_class == "mono":
                    mols_mono_glyconmr.append(mol)
                elif saccharide_class == "di":
                    mols_di_glyconmr.append(mol)
                elif saccharide_class == "tri":
                    mols_tri_glyconmr.append(mol)
                elif saccharide_class == "tetra":
                    mols_tetra_glyconmr.append(mol)
                elif saccharide_class == "oligo":
                    mols_oligo_glyconmr.append(mol)
                else:
                    print(f"ERROR: unknown mol saccharide class for GlycoNMR source - not in mono, di, tri, tetra, or oligo. Aborting!")
                    sys.exit()
            elif source == "CSDB":
                if saccharide_class == "mono":
                    mols_mono_csdb.append(mol)
                elif saccharide_class == "di":
                    mols_di_csdb.append(mol)
                elif saccharide_class == "tri":
                    mols_tri_csdb.append(mol)
                elif saccharide_class == "tetra":
                    mols_tetra_csdb.append(mol)
                elif saccharide_class == "oligo":
                    mols_oligo_csdb.append(mol)
                elif saccharide_class == "poly":
                    mols_poly_csdb.append(mol)
                else:
                    print(f"ERROR: unknown mol saccharide class for CSDB source - not in mono, di, tri, tetra, oligo, or poly. Aborting!")
                    sys.exit()
            else:
                print(f"ERROR: unknown mol source - not in CASPER, GlycoNMR, or CSDB. Aborting!")
                sys.exit()

    if not os.path.exists("roots/carbon/mono_casper"):  
        os.makedirs("roots/carbon/mono_casper")
    if not os.path.exists("roots/hydrogen/mono_casper"):  
        os.makedirs("roots/hydrogen/mono_casper")              
    mols_mono_carbon_d_casper = list(Carbons13C("roots/carbon/mono_casper", mols_mono_casper))
    mols_mono_hydrogen_d_casper = list(Hydrogens1H("roots/hydrogen/mono_casper", mols_mono_casper))

    if not os.path.exists("roots/carbon/di_casper"):  
        os.makedirs("roots/carbon/di_casper")
    if not os.path.exists("roots/hydrogen/di_casper"):  
        os.makedirs("roots/hydrogen/di_casper")              
    mols_di_carbon_d_casper = list(Carbons13C("roots/carbon/di_casper", mols_di_casper))
    mols_di_hydrogen_d_casper = list(Hydrogens1H("roots/hydrogen/di_casper", mols_di_casper))

    if not os.path.exists("roots/carbon/tri_casper"):  
        os.makedirs("roots/carbon/tri_casper")
    if not os.path.exists("roots/hydrogen/tri_casper"):  
        os.makedirs("roots/hydrogen/tri_casper")              
    mols_tri_carbon_d_casper = list(Carbons13C("roots/carbon/tri_casper", mols_tri_casper))
    mols_tri_hydrogen_d_casper = list(Hydrogens1H("roots/hydrogen/tri_casper", mols_tri_casper))

    if not os.path.exists("roots/carbon/mono_glyconmr"):  
        os.makedirs("roots/carbon/mono_glyconmr")
    if not os.path.exists("roots/hydrogen/mono_glyconmr"):  
        os.makedirs("roots/hydrogen/mono_glyconmr")              
    mols_mono_carbon_d_glyconmr = list(Carbons13C("roots/carbon/mono_glyconmr", mols_mono_glyconmr))
    mols_mono_hydrogen_d_glyconmr = list(Hydrogens1H("roots/hydrogen/mono_glyconmr", mols_mono_glyconmr))

    if not os.path.exists("roots/carbon/di_glyconmr"):  
        os.makedirs("roots/carbon/di_glyconmr")
    if not os.path.exists("roots/hydrogen/di_glyconmr"):  
        os.makedirs("roots/hydrogen/di_glyconmr")              
    mols_di_carbon_d_glyconmr = list(Carbons13C("roots/carbon/di_glyconmr", mols_di_glyconmr))
    mols_di_hydrogen_d_glyconmr = list(Hydrogens1H("roots/hydrogen/di_glyconmr", mols_di_glyconmr))

    if not os.path.exists("roots/carbon/tri_glyconmr"):  
        os.makedirs("roots/carbon/tri_glyconmr")
    if not os.path.exists("roots/hydrogen/tri_glyconmr"):  
        os.makedirs("roots/hydrogen/tri_glyconmr")              
    mols_tri_carbon_d_glyconmr = list(Carbons13C("roots/carbon/tri_glyconmr", mols_tri_glyconmr))
    mols_tri_hydrogen_d_glyconmr = list(Hydrogens1H("roots/hydrogen/tri_glyconmr", mols_tri_glyconmr))

    if not os.path.exists("roots/carbon/tetra_glyconmr"):  
        os.makedirs("roots/carbon/tetra_glyconmr")
    if not os.path.exists("roots/hydrogen/tetra_glyconmr"):  
        os.makedirs("roots/hydrogen/tetra_glyconmr")              
    mols_tetra_carbon_d_glyconmr = list(Carbons13C("roots/carbon/tetra_glyconmr", mols_tetra_glyconmr))
    mols_tetra_hydrogen_d_glyconmr = list(Hydrogens1H("roots/hydrogen/tetra_glyconmr", mols_tetra_glyconmr))

    if not os.path.exists("roots/carbon/oligo_glyconmr"):  
        os.makedirs("roots/carbon/oligo_glyconmr")
    if not os.path.exists("roots/hydrogen/oligo_glyconmr"):  
        os.makedirs("roots/hydrogen/oligo_glyconmr")              
    mols_oligo_carbon_d_glyconmr = list(Carbons13C("roots/carbon/oligo_glyconmr", mols_oligo_glyconmr))
    mols_oligo_hydrogen_d_glyconmr = list(Hydrogens1H("roots/hydrogen/oligo_glyconmr", mols_oligo_glyconmr))

    if not os.path.exists("roots/carbon/mono_csdb"):  
        os.makedirs("roots/carbon/mono_csdb")
    if not os.path.exists("roots/hydrogen/mono_csdb"):  
        os.makedirs("roots/hydrogen/mono_csdb")              
    mols_mono_carbon_d_csdb = list(Carbons13C("roots/carbon/mono_csdb", mols_mono_csdb))
    mols_mono_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/mono_csdb", mols_mono_csdb))

    if not os.path.exists("roots/carbon/di_csdb"):  
        os.makedirs("roots/carbon/di_csdb")
    if not os.path.exists("roots/hydrogen/di_csdb"):  
        os.makedirs("roots/hydrogen/di_csdb")              
    mols_di_carbon_d_csdb = list(Carbons13C("roots/carbon/di_csdb", mols_di_csdb))
    mols_di_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/di_csdb", mols_di_csdb))

    if not os.path.exists("roots/carbon/tri_csdb"):  
        os.makedirs("roots/carbon/tri_csdb")
    if not os.path.exists("roots/hydrogen/tri_csdb"):  
        os.makedirs("roots/hydrogen/tri_csdb")              
    mols_tri_carbon_d_csdb = list(Carbons13C("roots/carbon/tri_csdb", mols_tri_csdb))
    mols_tri_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/tri_csdb", mols_tri_csdb))

    if not os.path.exists("roots/carbon/tetra_csdb"):  
        os.makedirs("roots/carbon/tetra_csdb")
    if not os.path.exists("roots/hydrogen/tetra_csdb"):  
        os.makedirs("roots/hydrogen/tetra_csdb")              
    mols_tetra_carbon_d_csdb = list(Carbons13C("roots/carbon/tetra_csdb", mols_tetra_csdb))
    mols_tetra_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/tetra_csdb", mols_tetra_csdb))

    if not os.path.exists("roots/carbon/oligo_csdb"):  
        os.makedirs("roots/carbon/oligo_csdb")
    if not os.path.exists("roots/hydrogen/oligo_csdb"):  
        os.makedirs("roots/hydrogen/oligo_csdb")              
    mols_oligo_carbon_d_csdb = list(Carbons13C("roots/carbon/oligo_csdb", mols_oligo_csdb))
    mols_oligo_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/oligo_csdb", mols_oligo_csdb))

    if not os.path.exists("roots/carbon/poly_csdb"):  
        os.makedirs("roots/carbon/poly_csdb")
    if not os.path.exists("roots/hydrogen/poly_csdb"):  
        os.makedirs("roots/hydrogen/poly_csdb")              
    mols_poly_carbon_d_csdb = list(Carbons13C("roots/carbon/poly_csdb", mols_poly_csdb))
    mols_poly_hydrogen_d_csdb = list(Hydrogens1H("roots/hydrogen/poly_csdb", mols_poly_csdb))

    if not os.path.exists("conformations"):  
        os.makedirs("conformations")

    print(f"Creating CASPER conformers...")

    create_conformer(mols_mono_casper, num_confs)
    create_conformer(mols_di_casper, num_confs)
    create_conformer(mols_tri_casper, num_confs)

    print(f"Creating GlycoNMR conformers...")

    create_conformer(mols_mono_glyconmr, num_confs)
    create_conformer(mols_di_glyconmr, num_confs)
    create_conformer(mols_tri_glyconmr, num_confs)
    create_conformer(mols_tetra_glyconmr, num_confs)
    create_conformer(mols_oligo_glyconmr, num_confs)

    print(f"Creating CSDB conformers...")

    create_conformer(mols_mono_csdb, num_confs)
    create_conformer(mols_di_csdb, num_confs)
    create_conformer(mols_tri_csdb, num_confs)
    create_conformer(mols_tetra_csdb, num_confs)
    create_conformer(mols_oligo_csdb, num_confs)
    create_conformer(mols_poly_csdb, num_confs)

    for fold in range(0, 9 + 1):
        print(f"Building fold {fold} datasets...")
        train_data_carbon_mo_casper, test_data_carbon_mo_casper = train_test_split(mols_mono_carbon_d_casper, fold)
        train_data_carbon_di_casper, test_data_carbon_di_casper = train_test_split(mols_di_carbon_d_casper, fold)
        train_data_carbon_tri_casper, test_data_carbon_tri_casper = train_test_split(mols_tri_carbon_d_casper, fold)

        train_data_hydrogen_mo_casper, test_data_hydrogen_mo_casper = train_test_split(mols_mono_hydrogen_d_casper, fold)
        train_data_hydrogen_di_casper, test_data_hydrogen_di_casper = train_test_split(mols_di_hydrogen_d_casper, fold)
        train_data_hydrogen_tri_casper, test_data_hydrogen_tri_casper = train_test_split(mols_tri_hydrogen_d_casper, fold)

        train_data_carbon_mo_glyconmr, test_data_carbon_mo_glyconmr = train_test_split(mols_mono_carbon_d_glyconmr, fold)
        train_data_carbon_di_glyconmr, test_data_carbon_di_glyconmr = train_test_split(mols_di_carbon_d_glyconmr, fold)
        train_data_carbon_tri_glyconmr, test_data_carbon_tri_glyconmr = train_test_split(mols_tri_carbon_d_glyconmr, fold)
        train_data_carbon_tetra_glyconmr, test_data_carbon_tetra_glyconmr = train_test_split(mols_tetra_carbon_d_glyconmr, fold)
        train_data_carbon_oligo_glyconmr, test_data_carbon_oligo_glyconmr = train_test_split(mols_oligo_carbon_d_glyconmr, fold)

        train_data_hydrogen_mo_glyconmr, test_data_hydrogen_mo_glyconmr = train_test_split(mols_mono_hydrogen_d_glyconmr, fold)
        train_data_hydrogen_di_glyconmr, test_data_hydrogen_di_glyconmr = train_test_split(mols_di_hydrogen_d_glyconmr, fold)
        train_data_hydrogen_tri_glyconmr, test_data_hydrogen_tri_glyconmr = train_test_split(mols_tri_hydrogen_d_glyconmr, fold)
        train_data_hydrogen_tetra_glyconmr, test_data_hydrogen_tetra_glyconmr = train_test_split(mols_tetra_hydrogen_d_glyconmr, fold)
        train_data_hydrogen_oligo_glyconmr, test_data_hydrogen_oligo_glyconmr = train_test_split(mols_oligo_hydrogen_d_glyconmr, fold)

        train_data_carbon_mo_csdb, test_data_carbon_mo_csdb = train_test_split(mols_mono_carbon_d_csdb, fold)
        train_data_carbon_di_csdb, test_data_carbon_di_csdb = train_test_split(mols_di_carbon_d_csdb, fold)
        train_data_carbon_tri_csdb, test_data_carbon_tri_csdb = train_test_split(mols_tri_carbon_d_csdb, fold)
        train_data_carbon_tetra_csdb, test_data_carbon_tetra_csdb = train_test_split(mols_tetra_carbon_d_csdb, fold)
        train_data_carbon_oligo_csdb, test_data_carbon_oligo_csdb = train_test_split(mols_oligo_carbon_d_csdb, fold)
        train_data_carbon_poly_csdb, test_data_carbon_poly_csdb = train_test_split(mols_poly_carbon_d_csdb, fold)

        train_data_hydrogen_mo_csdb, test_data_hydrogen_mo_csdb = train_test_split(mols_mono_hydrogen_d_csdb, fold)
        train_data_hydrogen_di_csdb, test_data_hydrogen_di_csdb = train_test_split(mols_di_hydrogen_d_csdb, fold)
        train_data_hydrogen_tri_csdb, test_data_hydrogen_tri_csdb = train_test_split(mols_tri_hydrogen_d_csdb, fold)
        train_data_hydrogen_tetra_csdb, test_data_hydrogen_tetra_csdb = train_test_split(mols_tetra_hydrogen_d_csdb, fold)
        train_data_hydrogen_oligo_csdb, test_data_hydrogen_oligo_csdb = train_test_split(mols_oligo_hydrogen_d_csdb, fold)
        train_data_hydrogen_poly_csdb, test_data_hydrogen_poly_csdb = train_test_split(mols_poly_hydrogen_d_csdb, fold)


        train_data_carbon_casper = train_data_carbon_mo_casper + train_data_carbon_di_casper + train_data_carbon_tri_casper
        train_data_hydrogen_casper = train_data_hydrogen_mo_casper + train_data_hydrogen_di_casper + train_data_hydrogen_tri_casper

        train_data_carbon_glyconmr = train_data_carbon_mo_glyconmr + train_data_carbon_di_glyconmr + train_data_carbon_tri_glyconmr + train_data_carbon_tetra_glyconmr + train_data_carbon_oligo_glyconmr
        train_data_hydrogen_glyconmr = train_data_hydrogen_mo_glyconmr + train_data_hydrogen_di_glyconmr + train_data_hydrogen_tri_glyconmr + train_data_hydrogen_tetra_glyconmr + train_data_hydrogen_oligo_glyconmr

        train_data_carbon_csdb = train_data_carbon_mo_csdb + train_data_carbon_di_csdb + train_data_carbon_tri_csdb + train_data_carbon_tetra_csdb + train_data_carbon_oligo_csdb + train_data_carbon_poly_csdb
        train_data_hydrogen_csdb = train_data_hydrogen_mo_csdb + train_data_hydrogen_di_csdb + train_data_hydrogen_tri_csdb + train_data_hydrogen_tetra_csdb + train_data_hydrogen_oligo_csdb + train_data_hydrogen_poly_csdb

        train_data_carbon_mixed = train_data_carbon_casper + train_data_carbon_glyconmr + train_data_carbon_csdb
        train_data_hydrogen_mixed = train_data_hydrogen_casper + train_data_hydrogen_glyconmr + train_data_hydrogen_csdb

        train_data_carbon_casper = add_conformations_to_train_dataset(train_data_carbon_casper)
        train_data_hydrogen_casper = add_conformations_to_train_dataset(train_data_hydrogen_casper)

        train_data_carbon_mixed = add_conformations_to_train_dataset(train_data_carbon_mixed)
        train_data_hydrogen_mixed = add_conformations_to_train_dataset(train_data_hydrogen_mixed)

        if not os.path.exists(f"datasets-casper-fold{fold}"):
            os.makedirs(f"datasets-casper-fold{fold}")
        if not os.path.exists(f"datasets-mixed-fold{fold}"):
            os.makedirs(f"datasets-mixed-fold{fold}")

        with open(f"datasets-casper-fold{fold}/train_data_13C.pickle", 'wb') as handle:
            pickle.dump(train_data_carbon_casper, handle)
        with open(f"datasets-casper-fold{fold}/train_data_1H.pickle", 'wb') as handle:
            pickle.dump(train_data_hydrogen_casper, handle)

        with open(f"datasets-mixed-fold{fold}/train_data_13C.pickle", 'wb') as handle:
            pickle.dump(train_data_carbon_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/train_data_1H.pickle", 'wb') as handle:
            pickle.dump(train_data_hydrogen_mixed, handle)

        test_data_carbon_mo_mixed = test_data_carbon_mo_casper + test_data_carbon_mo_glyconmr + test_data_carbon_mo_csdb
        test_data_carbon_di_mixed = test_data_carbon_di_casper + test_data_carbon_di_glyconmr + test_data_carbon_di_csdb
        test_data_carbon_tri_mixed = test_data_carbon_tri_casper + test_data_carbon_tri_glyconmr + test_data_carbon_tri_csdb
        test_data_carbon_tetra_mixed = test_data_carbon_tetra_glyconmr + test_data_carbon_tetra_csdb
        test_data_carbon_oligo_mixed = test_data_carbon_oligo_glyconmr + test_data_carbon_oligo_csdb
        test_data_carbon_poly_mixed = [] + test_data_carbon_poly_csdb

        test_data_hydrogen_mo_mixed = test_data_hydrogen_mo_casper + test_data_hydrogen_mo_glyconmr + test_data_hydrogen_mo_csdb
        test_data_hydrogen_di_mixed = test_data_hydrogen_di_casper + test_data_hydrogen_di_glyconmr + test_data_hydrogen_di_csdb
        test_data_hydrogen_tri_mixed = test_data_hydrogen_tri_casper + test_data_hydrogen_tri_glyconmr + test_data_hydrogen_tri_csdb
        test_data_hydrogen_tetra_mixed = test_data_hydrogen_tetra_glyconmr + test_data_hydrogen_tetra_csdb
        test_data_hydrogen_oligo_mixed = test_data_hydrogen_oligo_glyconmr + test_data_hydrogen_oligo_csdb
        test_data_hydrogen_poly_mixed = [] + test_data_hydrogen_poly_csdb

        test_data_carbon_mo_casper = add_conformations_to_test_dataset(test_data_carbon_mo_casper)
        test_data_carbon_di_casper = add_conformations_to_test_dataset(test_data_carbon_di_casper)
        test_data_carbon_tri_casper = add_conformations_to_test_dataset(test_data_carbon_tri_casper)

        test_data_hydrogen_mo_casper = add_conformations_to_test_dataset(test_data_hydrogen_mo_casper)
        test_data_hydrogen_di_casper = add_conformations_to_test_dataset(test_data_hydrogen_di_casper)
        test_data_hydrogen_tri_casper = add_conformations_to_test_dataset(test_data_hydrogen_tri_casper)

        test_data_carbon_mo_mixed = add_conformations_to_test_dataset(test_data_carbon_mo_mixed)
        test_data_carbon_di_mixed = add_conformations_to_test_dataset(test_data_carbon_di_mixed)
        test_data_carbon_tri_mixed = add_conformations_to_test_dataset(test_data_carbon_tri_mixed)
        test_data_carbon_tetra_mixed = add_conformations_to_test_dataset(test_data_carbon_tetra_mixed)
        test_data_carbon_oligo_mixed = add_conformations_to_test_dataset(test_data_carbon_oligo_mixed)
        test_data_carbon_poly_mixed = add_conformations_to_test_dataset(test_data_carbon_poly_mixed)

        test_data_hydrogen_mo_mixed = add_conformations_to_test_dataset(test_data_hydrogen_mo_mixed)
        test_data_hydrogen_di_mixed = add_conformations_to_test_dataset(test_data_hydrogen_di_mixed)
        test_data_hydrogen_tri_mixed = add_conformations_to_test_dataset(test_data_hydrogen_tri_mixed)
        test_data_hydrogen_tetra_mixed = add_conformations_to_test_dataset(test_data_hydrogen_tetra_mixed)
        test_data_hydrogen_oligo_mixed = add_conformations_to_test_dataset(test_data_hydrogen_oligo_mixed)
        test_data_hydrogen_poly_mixed = add_conformations_to_test_dataset(test_data_hydrogen_poly_mixed)

        with open(f"datasets-casper-fold{fold}/test_data_13C_mo.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_mo_casper, handle)
        with open(f"datasets-casper-fold{fold}/test_data_13C_di.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_di_casper, handle)
        with open(f"datasets-casper-fold{fold}/test_data_13C_tri.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_tri_casper, handle)
        with open(f"datasets-casper-fold{fold}/test_data_1H_mo.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_mo_casper, handle)
        with open(f"datasets-casper-fold{fold}/test_data_1H_di.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_di_casper, handle)
        with open(f"datasets-casper-fold{fold}/test_data_1H_tri.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_tri_casper, handle)

        with open(f"datasets-mixed-fold{fold}/test_data_13C_mo.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_mo_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_13C_di.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_di_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_13C_tri.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_tri_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_13C_tetra.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_tetra_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_13C_oligo.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_oligo_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_13C_poly.pickle", 'wb') as handle:
            pickle.dump(test_data_carbon_poly_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_mo.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_mo_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_di.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_di_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_tri.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_tri_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_tetra.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_tetra_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_oligo.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_oligo_mixed, handle)
        with open(f"datasets-mixed-fold{fold}/test_data_1H_poly.pickle", 'wb') as handle:
            pickle.dump(test_data_hydrogen_poly_mixed, handle)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data generation script')
    parser.add_argument('--data_path', type=str, default="data/dataset/combined.sdf", help='Path to dataset')
    parser.add_argument('--num_confs', type=int, default=100, help='Number of conformations to generate')

    args = parser.parse_args()
    main(args.data_path, args.num_confs)
