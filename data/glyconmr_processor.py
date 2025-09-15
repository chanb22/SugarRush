# GlycoNMR Processor for GeqShift/SugarRush
# Channing Bellamy (BLLCHA013)
# University of Cape Town

import csv
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import random

def read_shifts_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize/strip
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            # Coerce residual to int if possible
            try:
                row['residual'] = int(float(row['residual']))
            except Exception:
                pass
            # Coerce labels to float
            try:
                row['labels'] = float(row['labels'])
            except Exception:
                row['labels'] = None
            rows.append(row)
    return rows

def main():
    mols_mono = []
    mols_di = []
    mols_tri = []
    mols_tetra = []
    mols_oligo = []
    mols_poly = []

    for entry in os.scandir("glyconmr/GlycoNMR.Exp_processed"):
        name = entry.name[:entry.name.find(".")]

        # Read PDB
        try:
            with open(f"glyconmr/GlycoNMR.Exp_raw_files/{name}.pdb", 'r') as fh:
                pdb_block = fh.read()
        except:
            continue

        mol = Chem.MolFromPDBBlock(
            pdb_block,
            sanitize=True,
            removeHs=False,
            proximityBonding=True
        )
        if mol is None:
            continue

        # Standardize / assign stereochemistry
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        residues = []

        # Read shifts CSV and index
        rows = read_shifts_csv(f"glyconmr/GlycoNMR.Exp_processed/{name}.pdb.csv")
        
        for item in rows:
            if not item['residual'] in residues:
                residues.append(item['residual'])

        # Remove Hs for carbon indexing
        mol_noH = Chem.RemoveHs(mol, implicitOnly=False)
        Chem.AssignAtomChiralTagsFromStructure(mol_noH)

        # Map from mol_noH atom index to mol atom index
        map_noH_to_full = {}
        for a_noH in mol_noH.GetAtoms():
            pos_noH = a_noH.GetIdx()
            symbol = a_noH.GetSymbol()
            # Find corresponding atom in full mol by matching coordinates or PDB info
            info_noH = a_noH.GetPDBResidueInfo()
            match_idx = None
            for a_full in mol.GetAtoms():
                if a_full.GetSymbol() != symbol:
                    continue
                info_full = a_full.GetPDBResidueInfo()
                if info_noH and info_full:
                    if (info_noH.GetResidueName().strip().upper() == info_full.GetResidueName().strip().upper() and
                        info_noH.GetResidueNumber() == info_full.GetResidueNumber() and
                        ''.join(info_noH.GetName().split()).upper() == ''.join(info_full.GetName().split()).upper()):
                        match_idx = a_full.GetIdx()
                        break
            if match_idx is None:
                raise ValueError(f"Could not map noH atom {pos_noH} to full mol")
            map_noH_to_full[pos_noH] = match_idx

        # Build spectrum dicts
        c_spectrum = {}
        h_spectrum = {}
        for idx_noH, idx_full in map_noH_to_full.items():
            atom_full = mol.GetAtomWithIdx(idx_full)
            if atom_full.GetSymbol() == "C" and atom_full.HasProp("SHIFT_13C_PPM"):
                c_spectrum[idx_noH] = atom_full.GetDoubleProp("SHIFT_13C_PPM")
                # Collect H shifts for H atoms bound to this C
                hvals = []
                for nbr in atom_full.GetNeighbors():
                    if nbr.GetSymbol() == "H" and nbr.HasProp("SHIFT_1H_PPM"):
                        hvals.append(nbr.GetDoubleProp("SHIFT_1H_PPM"))
                if hvals:
                    # If multiple, you could store all, but your format seems to want a single
                    # I'll take the first for now
                    h_spectrum[idx_noH] = [hvals[0], 0]
        
        for key in c_spectrum:
            if key not in h_spectrum.keys():
                h_spectrum[key] = [-1.0, 0.0]
        
        num_unknown_c13 = 0
        num_unknown_1h = 0
        for shift in c_spectrum.values():
            if shift == -1.0:
                num_unknown_c13 += 1
        for shift in h_spectrum.values():
            if shift[0] == -1.0:
                num_unknown_1h += 1

        if ((num_unknown_c13 / (len(c_spectrum.values()) + 1)) >= 0.2) or ((num_unknown_1h / (len(h_spectrum.values()) + 1)) >= 0.2):
            print(f"{num_unknown_c13} / {len(c_spectrum.values()) + 1} | {num_unknown_1h} / {len(h_spectrum.values()) + 1}")
            continue

        delete = {k: v for k, v in c_spectrum.items() if v == -1}
        for key in delete:
            c_spectrum.pop(key)
        delete = {k: v for k, v in h_spectrum.items() if v[0] == -1}
        for key in delete:
            h_spectrum.pop(key)

        # Store as SD tags
        mol.SetProp("13C Spectrum", str(c_spectrum))
        mol.SetProp("1H Spectrum", str(h_spectrum))

        # Preserve original PDB title if present
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if not title:
            # Build a short name from residues
            residues = defaultdict(set)
            for a in mol.GetAtoms():
                info = a.GetPDBResidueInfo()
                if info:
                    residues[info.GetResidueName().strip().upper()].add(info.GetResidueNumber())
            parts = [f"{rn}:{','.join(str(x) for x in sorted(residues[rn]))}" for rn in sorted(residues)]
            mol.SetProp("_Name", " + ".join(parts) if parts else "Converted_From_PDB")
        
        mol.SetProp("_Name", name)

        mol = Chem.RemoveHs(mol, implicitOnly=False)

        if len(residues) == 1:
            mols_mono.append(mol)
        if len(residues) == 2:
            mols_di.append(mol)
        if len(residues) == 3:
            mols_tri.append(mol)
        if len(residues) == 4:
            mols_tetra.append(mol)
        if len(residues) > 4 and len(residues) <= 10:
            mols_oligo.append(mol)
        if len(residues) > 10:
            mols_poly.append(mol)
        print(f"generated mol {name}")

    random.shuffle(mols_mono)
    random.shuffle(mols_di)
    random.shuffle(mols_tri)
    random.shuffle(mols_tetra)
    random.shuffle(mols_oligo)
    random.shuffle(mols_poly)

    # Write SDF
    with Chem.SDWriter(f"glyconmr/glyconmr_mono.sdf") as w:
        for mol in mols_mono:
            w.write(mol)

    with Chem.SDWriter(f"glyconmr/glyconmr_dimer.sdf") as w:
        for mol in mols_di:
            w.write(mol)

    with Chem.SDWriter(f"glyconmr/glyconmr_trimer.sdf") as w:
        for mol in mols_tri:
            w.write(mol)

    with Chem.SDWriter(f"glyconmr/glyconmr_tetra.sdf") as w:
        for mol in mols_tetra:
            w.write(mol)

    with Chem.SDWriter(f"glyconmr/glyconmr_oligomer.sdf") as w:
        for mol in mols_oligo:
            w.write(mol)

    with Chem.SDWriter(f"glyconmr/glyconmr_polymer.sdf") as w:
        for mol in mols_poly:
            w.write(mol)

    print("Done.")

if __name__ == "__main__":
    main()
