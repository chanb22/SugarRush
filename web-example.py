# Example Web API and website for SugarRush
# Channing Bellamy (BLLCHA013)
# University of Cape Town
# Uses work from University of Stockholm: GeqShift paper arXiv:2311.12657
# Uses 3Dmol.js: https://doi.org/10.1093/bioinformatics/btu829

from flask import Flask, request, jsonify, Response, render_template, send_file
from rdkit.Chem import rdmolfiles, Draw
from io import BytesIO

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model.model import O3Transformer
from model.norms import EquivariantLayerNorm
from e3nn import o3

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.nn import radius_graph
import numpy as np
import pickle

torch.manual_seed(42)

print(f"INFO: loading training data...")

with open("train_data_13C.pickle", 'rb') as handle:
    train_data_13C = pickle.load(handle)
with open("train_data_1H.pickle", 'rb') as handle:
    train_data_1H = pickle.load(handle)

mean_13C = 0
mean_1H = 0
std_13C = 0
std_1H = 0

def derive_mean_and_std(train_data):
    nmrs = []
    for data in train_data:
        nmr_true =  data.x[:,-1]
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)    
        nmrs.append(nmr_true[mask])
    nmrs = torch.cat(nmrs)
    return nmrs.mean(), nmrs.std()

print(f"INFO: deriving means and standard deviations from training data...")

mean_13C, std_13C = derive_mean_and_std(train_data=train_data_13C)
mean_1H, std_1H = derive_mean_and_std(train_data=train_data_1H)

print(f"INFO: 13C/1H means are {mean_13C}/{mean_1H}, 13C/1H standard deviations are {std_13C}/{std_1H}")

del(train_data_13C)
del(train_data_1H)

types = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br':9}
bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: using torch device {device} for inference...")

model_13C = O3Transformer(norm = EquivariantLayerNorm, n_input = 128,n_node_attr = 128,n_output =128, 
                                      irreps_hidden = o3.Irreps("64x0e + 32x1o + 8x2e"),n_layers = 7)
model_1H = O3Transformer(norm = EquivariantLayerNorm, n_input = 128,n_node_attr = 128,n_output =128, 
                                      irreps_hidden = o3.Irreps("64x0e + 32x1o + 8x2e"),n_layers = 7)

print(f"INFO: moving models to device...")
model_13C.to(device=device)
model_1H.to(device=device)

print(f"INFO: loading state dicts for models...")
model_13C.load_state_dict(torch.load("checkpoint-13C-casper.pkl", map_location=device))
model_13C.eval()
model_1H.load_state_dict(torch.load("checkpoint-1H-casper.pkl", map_location=device))
model_1H.eval()

print(f"INFO: creating Flask app...")
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mol2d", methods=["POST"])
def mol2d():
    content = request.json
    smiles = content.get("smiles", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "failed to process SMILES"}), 400
    
    img = Draw.MolToImage(mol, size=(500, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/mol3d", methods=["POST"])
def mol3d():
    content = request.json
    if "smiles" not in content:
        return jsonify({"error": "missing SMILES string"}), 400

    smiles = content["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    sdf = rdmolfiles.MolToMolBlock(mol)
    return Response(sdf, mimetype="chemical/x-mdl-sdfile")

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    if "smiles" not in content:
        return jsonify({"error": "missing SMILES string"}), 400
    smiles = content["smiles"]
    print(f"INFO: constructing molecule from SMILES...")
    try:
        mol = Chem.MolFromSmiles(SMILES=smiles)
    except:
        return jsonify({"error": "failed to process SMILES"}), 400

    type_idx_list = []
    num_hs = []
    carbon_indices = []

    for atom in mol.GetAtoms():
        type_idx_list.append(types[atom.GetSymbol()])
        num_hs.append(atom.GetTotalNumHs())
        if atom.GetAtomicNum() == 6:
            carbon_indices.append(atom.GetIdx())

    type_idx = torch.tensor(type_idx_list, dtype=torch.float).reshape(-1,1)
    num_hs = torch.tensor(num_hs, dtype=torch.float).reshape(-1,1)

    x = torch.cat([type_idx, num_hs], dim=-1)

    row, col = [], []
    bond_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start]
        col += [end]
        bond_attr +=  [bonds[bond.GetBondType()]]
    bond_attr = torch.tensor(bond_attr, dtype=torch.long).flatten()
    edge_index_b = torch.tensor([row, col], dtype=torch.long)
    edge_index_b, bond_attr = to_undirected(edge_index_b,bond_attr)

    positions = torch.zeros((x.size(0), 3), dtype=torch.float)

    edge_index_r = radius_graph(positions, r=1000.)
    edge_index_r = to_undirected(edge_index_r)
    edge_attr_r = torch.zeros(edge_index_r.size(1))
    edge_index = torch.column_stack([edge_index_b,edge_index_r])
    edge_attr = torch.cat([bond_attr, edge_attr_r])
    edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='max')

    data = Data()
    data.x = x
    data.edge_attr = edge_attr
    data.edge_index = edge_index
    data.pos = positions

    params = Chem.rdDistGeom.ETKDGv3()
    params.pruneRmsThresh = 0.01
    params.numThreads = 0
    params.maxIterations = 10000
    params.randomSeed = 42

    Chem.SanitizeMol(mol=mol)
    mol = Chem.AddHs(mol=mol)

    print(f"INFO: embedding 100 conformations...")
    em = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=100*2, params=params)
    ps = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')

    energies = []
    for conf in mol.GetConformers():
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, ps, confId=conf.GetId())

        if isinstance(ff, type(None)):
            print(f"ERROR: failed to get molecule force field!")
            continue
        energy = ff.CalcEnergy()
        energies.append(energy)

    mol = Chem.RemoveHs(mol=mol)
    if em == -1:
        print(f"ERROR: failed to generate conformations!")

    energies = np.array(energies)
    ind = energies.argsort()[:100]
    energies = energies[ind]
    conformations = []
    for i, c in enumerate(mol.GetConformers()):
        if i not in ind:
            continue
        xyz = c.GetPositions()
        conformations.append(xyz)

    if len(conformations) < 100:
         print(f"ERROR: failed to generate enough conformers!")

    def get_cut_off_graph(edge_index, edge_attr, p, cut_off = 6.0):
        row, col = edge_index
        dist = torch.sqrt(torch.sum((p[row]- p[col])**2, dim = 1))
        mask = dist <= cut_off
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
        edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce  = "max")
        return edge_index, edge_attr

    new_dataset = []

    k = 0
    for conformer in conformations:
        new_data = data.clone()
        new_data.pos = torch.from_numpy(conformer).float()
        edge_index, edge_attr = get_cut_off_graph(new_data.edge_index, new_data.edge_attr, new_data.pos ,cut_off=6.)
        new_data.edge_index = edge_index
        new_data.edge_attr = edge_attr
        new_dataset.append(new_data)
        k += 1
    if len(new_dataset) != 100:
         print(f"ERROR: unexpected number of conformers in dataset!")

    c_mask = data.x[:,0] == 2.0
    mask = c_mask

    n_nmr = torch.sum(mask).item()
    n_mols = k
    loader = DataLoader(new_dataset, batch_size=len(new_dataset), shuffle=False)

    N = n_mols
    N_nmr = n_nmr
    print(f"INFO: running inference...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True) 
            c_mask = data.x[:,0] == 2.0
            mask = c_mask
            out_13C = model_13C(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                        edge_index = data.edge_index, edge_attr = data.edge_attr.long(), batch = data.batch)
            out_1H = model_1H(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                        edge_index = data.edge_index, edge_attr = data.edge_attr.long(), batch = data.batch)
            out_masked_13C = out_13C[mask]*std_13C + mean_13C
            out_masked_1H = out_1H[mask]*std_1H + mean_1H

            out_13C = (out_masked_13C.reshape(N,N_nmr).T).mean(dim = 1)
            out_1H = (out_masked_1H.reshape(N,N_nmr).T).mean(dim = 1)
        print(f"13C predicted peaks: {out_13C.detach().flatten()}")
        print(f"1H predicted peaks: {out_1H.detach().flatten()}")

        # map each predicted 13C shift to a carbon atom index
        pred_13C = out_13C.detach().cpu().flatten().tolist()
        pred_13C_mapped = [
            {"atom_index": idx, "shift": shift}
            for idx, shift in zip(carbon_indices, pred_13C)
        ]

        pred_1H = out_1H.detach().cpu().flatten().tolist()
        pred_1H_mapped = [
            {"atom_index": i, "shift": shift}
            for i, shift in zip(carbon_indices, pred_1H)
        ]

        return jsonify({
            "smiles": smiles,
            "atoms": list(range(N_nmr)),
            "predicted_13C": pred_13C_mapped,
            "predicted_1H": pred_1H_mapped
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
