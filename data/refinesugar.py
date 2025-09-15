# Sugar Refinery
# CSDB Dump Validation and Convergence Program
# Converges and validates PDB, PDBG, and MOL files obtained from CSDB.
# Channing Bellamy (BLLCHA013)
# University of Cape Town

import os
import sweetener

BASE_DOWNLOAD_PATH = "csdb"
PDB_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/pdb"
PDBG_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/pdbg"
MOL_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/mol"

if os.path.exists(BASE_DOWNLOAD_PATH):
    if os.path.isdir(BASE_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {BASE_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {BASE_DOWNLOAD_PATH} is not a directory.")
else:
    print(f"WARNING: creating directory {BASE_DOWNLOAD_PATH}")
    os.mkdir(BASE_DOWNLOAD_PATH)

if os.path.exists(PDB_DOWNLOAD_PATH):
    if os.path.isdir(PDB_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {PDB_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {PDB_DOWNLOAD_PATH} is not a directory.")
else:
    print(f"WARNING: creating directory {PDB_DOWNLOAD_PATH}")
    os.mkdir(PDB_DOWNLOAD_PATH)

if os.path.exists(PDBG_DOWNLOAD_PATH):
    if os.path.isdir(PDBG_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {PDBG_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {PDBG_DOWNLOAD_PATH} is not a directory.")
else:
    print(f"WARNING: creating directory {PDBG_DOWNLOAD_PATH}")
    os.mkdir(PDBG_DOWNLOAD_PATH)

if os.path.exists(MOL_DOWNLOAD_PATH):
    if os.path.isdir(MOL_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {MOL_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {MOL_DOWNLOAD_PATH} is not a directory.")
else:
    print(f"WARNING: creating directory {MOL_DOWNLOAD_PATH}")
    os.mkdir(MOL_DOWNLOAD_PATH)

sweetener.delete_invalid_pdb(PDB_DOWNLOAD_PATH)
sweetener.delete_invalid_pdbg(PDBG_DOWNLOAD_PATH)
sweetener.delete_invalid_mol(MOL_DOWNLOAD_PATH)

pdb_ids_existing = []
pdbg_ids_existing = []

print(f"INFO: discovering existing PDB IDs...")
for entry in os.scandir(PDB_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('.')]
        if csdb_id not in pdb_ids_existing:
            pdb_ids_existing.append(csdb_id)
print(f"INFO: we have {len(pdb_ids_existing)} PDB IDs.")

print(f"INFO: discovering existing PDBG IDs...")
for entry in os.scandir(PDBG_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('.')]
        if csdb_id not in pdbg_ids_existing:
            pdbg_ids_existing.append(csdb_id)
print(f"INFO: we have {len(pdbg_ids_existing)} PDBG IDs.")

mol_ids_existing = []

print(f"INFO: discovering existing MOL IDs...")
for entry in os.scandir(MOL_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('-')]
        if csdb_id not in mol_ids_existing:
            mol_ids_existing.append(csdb_id)
print(f"INFO: we have {len(mol_ids_existing)} MOL IDs.")

print(f"INFO: cross-checking file existence...")

delete_pdb = []
delete_pdbg = []
delete_mol = []

for item in pdb_ids_existing:
    if ((item not in pdbg_ids_existing) or (item not in mol_ids_existing)) and (item not in delete_pdb):
        print(f"WARNING: found dangling PDB file ID {item}")
        delete_pdb.append(item)

for item in pdbg_ids_existing:
    if ((item not in pdb_ids_existing) or (item not in mol_ids_existing)) and (item not in delete_pdbg):
        print(f"WARNING: found dangling PDBG file ID {item}")
        delete_pdbg.append(item)

for item in mol_ids_existing:
    if ((item not in pdb_ids_existing) or (item not in pdbg_ids_existing)) and (item not in delete_mol):
        print(f"WARNING: found dangling MOL file ID {item}")
        delete_mol.append(item)
    if ((item in delete_pdb) or (item in delete_pdbg)) and (item not in delete_mol):
        print(f"WARNING: found dangling MOL file ID {item}")
        delete_mol.append(item)

print(f"WARNING: found {len(delete_pdb)} dangling PDB IDs, {len(delete_pdbg)} dangling PDBG IDs, and {len(delete_mol)} dangling MOL IDs.")

for entry in os.scandir(PDB_DOWNLOAD_PATH):
    if entry.name[:entry.name.find('.')] in delete_pdb:
        os.remove(entry.path)
        print(f"WARNING: deleted {entry.path}")
for entry in os.scandir(PDBG_DOWNLOAD_PATH):
    if entry.name[:entry.name.find('.')] in delete_pdbg:
        os.remove(entry.path)
        print(f"WARNING: deleted {entry.path}")
for entry in os.scandir(MOL_DOWNLOAD_PATH):
    if entry.name[:entry.name.find('-')] in delete_mol:
        os.remove(entry.path)
        print(f"WARNING: deleted {entry.path}")

print(f"INFO: sugar refined.")
