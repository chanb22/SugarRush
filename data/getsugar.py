# REStMORE
# CSDB Dump Fetch Program
# Scans a CSDB Dump file and fetches PDB/PDBG and MOL files from CSDB API
# Channing Bellamy (BLLCHA013)
# University of Cape Town

import os
import requests
import sweetener

DUMP_PATH = "csdb_nmr_data_2025Apr12_nosim.txt"
BASE_DOWNLOAD_PATH = "csdb"
PDB_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/pdb"
PDBG_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/pdbg"
MOL_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/mol"

if not os.path.isfile(DUMP_PATH):
    print(f"ERROR: {DUMP_PATH} must be a file that exists.")

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

print(f"INFO: loading CSDB dump file {DUMP_PATH}...")
dump = open(DUMP_PATH)
dump_lines = dump.readlines()
dump_data = sweetener.load_dump(dump_lines)
del(dump_lines)
dump.close()

pdb_ids_existing = []
pdbg_ids_existing = []

print(f"INFO: discovering existing PDB files...")
for entry in os.scandir(PDB_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('.')]
        pdb_ids_existing.append(csdb_id)
print(f"INFO: we have {len(pdb_ids_existing)} PDB IDs out of {len(dump_data)}")

print(f"INFO: discovering existing PDBG files...")
for entry in os.scandir(PDBG_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('.')]
        pdbg_ids_existing.append(csdb_id)
print(f"INFO: we have {len(pdbg_ids_existing)} PDBG IDs out of {len(dump_data)}")

pdb_download_queue = []
pdbg_download_queue = []

print(f"INFO: building PDB/PDBG download queues...")
for record in dump_data:
    csdb_id = record.get('CSDB_ID')
    if csdb_id not in pdb_ids_existing:
        pdb_download_queue.append(csdb_id)
    if csdb_id not in pdbg_ids_existing:
        pdbg_download_queue.append(csdb_id)
print(f"INFO: will fetch {len(pdb_download_queue)} PDB files from CSDB.")
print(f"INFO: will fetch {len(pdbg_download_queue)} PDBG files from CSDB.")

for item in pdb_download_queue:
    structure: str = ""
    for record in dump_data:
        if record.get('CSDB_ID') == item:
            structure = record.get('STRUCTURE')
            break
    if len(structure) <= 0:
        print(f"ERROR: failed to find CSDB ID {item} in dump, skipping...")
        break
    post_data = { 'csdb': structure,
                  'format': 'pdb' }
    try:
        print(f"INFO: ({pdb_download_queue.index(item)}/{len(pdb_download_queue)}) requesting PDB for {item} with structure {structure}...")
        response = requests.post(url="http://csdb.glycoscience.ru/database/core/convert_api.php", data=post_data, timeout=60)
        with open(f"{PDB_DOWNLOAD_PATH}/{item}.pdb", 'w') as file:
            file.write(response.text)
            print(f"INFO: saved PDB for {item} with structure {structure}.")
    except:
        print(f"ERROR: request failed for {item} with structure {structure}, skipping...")
        continue

for item in pdbg_download_queue:
    structure: str = ""
    for record in dump_data:
        if record.get('CSDB_ID') == item:
            structure = record.get('STRUCTURE')
            break
    if len(structure) <= 0:
        print(f"ERROR: failed to find CSDB ID {item} in dump, skipping...")
        break
    post_data = { 'csdb': structure,
                  'format': 'pdbg' }
    try:
        print(f"INFO: ({pdbg_download_queue.index(item)}/{len(pdbg_download_queue)}) requesting PDBG for {item} with structure {structure}...")
        response = requests.post(url="http://csdb.glycoscience.ru/database/core/convert_api.php", data=post_data, timeout=60)
        with open(f"{PDBG_DOWNLOAD_PATH}/{item}.pdb", 'w') as file:
            file.write(response.text)
            print(f"INFO: saved PDBG for {item} with structure {structure}.")
    except:
        print(f"ERROR: request failed for {item} with structure {structure}, skipping...")
        continue

mol_existing: str = []

print(f"INFO: discovering existing MOL files...")
for entry in os.scandir(MOL_DOWNLOAD_PATH):
    if entry.is_file():
        csdb_id = entry.name[:entry.name.find('.')]
        mol_existing.append(csdb_id)
print(f"INFO: we have {len(mol_existing)} MOL files.")

mol_available: list[tuple[str, str]] = []

print(f"INFO: finding available MOL file downloads...")
for entry in os.scandir(PDB_DOWNLOAD_PATH):
    if entry.is_file():
        file = open(entry.path)
        stereomer = 0
        for line in file.readlines():
            for item in line.split():
                index = item.find("?file=")
                if index > -1:
                    stereomer += 1
                    mol_available.append((f"{entry.name[:entry.name.find('.')]}-{stereomer}", item[index+6]))
print(f"INFO: found {len(mol_available)} MOL files avaiable for download.")

mol_download_queue: list[tuple[str, str]] = []

print(f"INFO: building MOL download queue...")

for item in mol_available:
    if item[0] not in mol_existing and item not in mol_download_queue:
        mol_download_queue.append(item)

for item in mol_download_queue:
    try:
        print(f"INFO: ({mol_download_queue.index(item)}/{len(mol_download_queue)}) requesting MOL file {item[0]}...")
        response = requests.get(url=f"http://csdb.glycoscience.ru/jsmol/mols/{item[1]}.mol", timeout=60)
        file = open(f"{MOL_DOWNLOAD_PATH}/{item[0]}.mol", 'w')
        file.write(response.text)
        file.close()
        print(f"INFO: saved MOL for {item[0]}.")
    except:
        print(f"ERROR: request failed for MOL file {item[0]}, skipping...")
        continue

print(f"INFO: you've got sugar.")
