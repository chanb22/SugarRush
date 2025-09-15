# CSDB Dump Processor Program
# Fitlers, and processes CSDB MOL/PDB/PDBG files into SDF files.
# Channing Bellamy (BLLCHA013)
# University of Cape Town

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

import os
import sys
import sweetener
import random

import re
from typing import Dict, List, Union, Tuple, Optional

Shift = Union[float, List[float], None]  # [low, diff] or None
Shifts = List[Shift]

# Patterns
# Matches values like "0.89", ".89", "64", "64.", etc.
VALUE_RE = re.compile(r'^\s*\d*\.\d+|\d+\.\d*|\d+\s*$')

# Matches multi-value ranges like "3.60-3.64-3.55", ".89-.92-.95"
MULTI_RE = re.compile(r'^\s*([\d\.]+(?:\s*-\s*[\d\.]+)+)\s*$')

# Matches range like "4.48 - 4.52" or ".48-?"
RANGE_RE = re.compile(r'^\s*([\d\.]+)\s*-\s*([\d\.]+|\?)\s*$')

# Matches flipped range like "? - .52"
RANGE_RE_FLIP = re.compile(r'^\s*(\?)\s*-\s*([\d\.]+)\s*$')

HEAD_RE = re.compile(r'^\s*#([^_\s]*)_([^\s]+)\s*(.*)$')

def parse_shift_token(tok: str) -> Shift:
    tok = tok.strip()
    if tok in ('-', '?', ''):
        return None

    # Multi-value average range (e.g., 3.60-3.64-3.55 or .89-.92-.95)
    m = MULTI_RE.match(tok)
    if m:
        try:
            parts = [float(x.strip()) for x in tok.split('-') if x.strip() and x != '?']
            if parts:
                spread = max(parts) - min(parts)
                return [min(parts), spread] # geqshift casper pipeline adds half of spread to min to get "average" or "middle" value already
        except ValueError:
            return None

    # Range like 4.48-4.52 or 4.48-?
    m = RANGE_RE.match(tok)
    if m:
        try:
            low = float(m.group(1))
            high_str = m.group(2)
            if high_str == '?':
                return [low, 0.0]
            high = float(high_str)
            return [low, high-low] # geqshift casper pipeline adds half of spread to min to get "average" or "middle" value already
        except ValueError:
            return None

    # Flipped range like ?-4.48
    m = RANGE_RE_FLIP.match(tok)
    if m:
        try:
            high = float(m.group(2))
            return [high, 0.0]
        except ValueError:
            return None

    # Single value (e.g., ".89", "64.", "64")
    if VALUE_RE.match(tok):
        try:
            return [float(tok), 0.0]
        except ValueError:
            return None

    print(f"ERROR: unknown token '{tok}'")
    return None

def parse_csdb_line(line: str) -> Dict[str, Shifts]:
    by_context: Dict[str, Shifts] = {}
    parts = [p.strip() for p in line.split('//')]
    for part in parts:
        if not part:
            continue
        m = HEAD_RE.match(part)
        if not m:
            continue
        context, residue, rest = m.group(1), m.group(2), m.group(3)
        tokens = rest.split()
        shifts: Shifts = [parse_shift_token(t) for t in tokens]
        key = f"#{context}_{residue}"
        by_context[key] = shifts
    return by_context

# Mapping parsing
ShiftPair = List[float]
ResShifts = List[ShiftPair]
C13Dict = Dict[str, ResShifts]        # key: residue tag like '#3,6_aDGalp'
H1Dict  = Dict[str, ResShifts]
IdxMap = Dict[int, str]      # atom_idx0 -> label like 'C1'
ResMaps = Dict[str, IdxMap]  # canonical_key -> IdxMap

def remove_percent_tags(s: str) -> str:
    return re.sub(r'_(?:[^_%]*)%\s*', '_', s)

def _canon_key(s: str) -> str:
    """
    Canonicalize residue names
    """
    s = remove_percent_tags(s)
    s = s.strip().replace(' ', '').replace('_', '').replace('-', '').replace('xX', '').replace('=', '').replace('0,0', '')
    return s

def to_pattern(key: str) -> str:
    """
    Turn a canonicalized key into a regex pattern.
    """
    canon = _canon_key(key)
    return '^' + re.escape(canon).replace(r'\?', '.?') + '$'

def parse_mapping_line(mapping_line: str, veto_atoms: set) -> ResMaps:
    """
    Parse the 'Mapping: ...' line
    """
    res_maps: ResMaps = {}
    # Find all blocks like "#... { a:b, c:d, ... }"
    for hdr, body in re.findall(r'#\s*([^{};]+?)\s*\{([^}]*)\}', mapping_line):
        raw_key = '#' + hdr.strip()
        key = _canon_key(raw_key)
        idx_map: IdxMap = {}
        for pair in body.split(','):
            pair = pair.strip()
            if not pair or ':' not in pair:
                continue
            i_str, label = pair.split(':', 1)
            idx0 = int(i_str.strip()) - 1  # convert to 0-index
            for index in veto_atoms:
                if idx0 >= index:
                    idx0 -= 1
            idx_map[idx0] = label.strip()
        res_maps[key] = idx_map
    return res_maps

def match_wildcard_key(query_key: str, spectrum_keys: Dict[str, Shifts]) -> Optional[Shifts]:
    query_key_canon = _canon_key(query_key).replace('?', '')

    # First try exact canonical match
    for k in spectrum_keys:
        if _canon_key(k) == query_key_canon:
            return spectrum_keys[k]

    # Try wildcard pattern matching
    for k in spectrum_keys:
        spectrum_key = to_pattern(_canon_key(k))
        if re.match(spectrum_key, query_key_canon):
            return spectrum_keys[k]
        
    if query_key_canon.endswith('p') or query_key_canon.endswith('f'):
        query_key_canon = query_key_canon[:-1]
    for k in spectrum_keys:
        spectrum_key = to_pattern(_canon_key(k))
        if re.match(spectrum_key, query_key_canon):
            return spectrum_keys[k]

    query_key_canon = query_key_canon.replace('p', '').replace('f', '')
    for k in spectrum_keys:
        spectrum_key = to_pattern(_canon_key(k))
        if re.match(spectrum_key, query_key_canon):
            return spectrum_keys[k]

    return None

def build_spectra(c13: C13Dict, h1: H1Dict, res_maps: ResMaps) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, ShiftPair]]]:
    """
    Returns:
      c13_spec: { atom_idx0 : 13C_value }  (sorted by atom index)
      h1_spec : { atom_idx0 : [1H_value, diff] } (sorted by atom index)
    """
    # Canonicalize all keys in spectrum dicts
    c13_can = { _canon_key(k): v for k, v in c13.items() }
    h1_can  = { _canon_key(k): v for k, v in h1.items() }

    c13_spec_temp: Dict[int, float] = {}
    h1_spec_temp : Dict[int, ShiftPair] = {}

    def resolve_shifts(key: str, spectra: Dict[str, Shifts]) -> Optional[Shifts]:
        if key in spectra:
            return spectra[key]
        return match_wildcard_key(key, spectra)

    for can_key, idx_map in res_maps.items():
        c_list = resolve_shifts(can_key, c13_can)
        h_list = resolve_shifts(can_key, h1_can)

        # Extract carbon atoms as (atom_idx0, label), e.g., (13, 'C1')
        carbon_atoms = [(idx0, label) for idx0, label in idx_map.items()
                        if label.startswith('C') and label[1:].isdigit()]
        carbon_atoms_sorted = sorted(carbon_atoms, key=lambda x: int(x[1][1:]))

        for i, (idx0, label) in enumerate(carbon_atoms_sorted):
            # C13 shift
            if c_list and i < len(c_list):
                c_val = c_list[i]
                if c_val is None:
                    continue
                c13_spec_temp[idx0] = float(c_val[0]) if isinstance(c_val, list) else float(c_val)

            # H1 shift
            if h_list and i < len(h_list):
                h_val = h_list[i]
                if h_val is None:
                    continue
                elif isinstance(h_val, float):
                    h_val = [h_val, 0.0]
                h1_spec_temp[idx0] = h_val

    c13_spec = dict(sorted(c13_spec_temp.items()))
    h1_spec  = dict(sorted(h1_spec_temp.items()))
    return c13_spec, h1_spec

DUMP_PATH = "csdb_nmr_data_2025Apr12_nosim.txt"
BASE_DOWNLOAD_PATH = "csdb"
MOL_DOWNLOAD_PATH = f"{BASE_DOWNLOAD_PATH}/mol"
BASE_SDF_OUTPUT_PATH = "sdf"

if not os.path.isfile(DUMP_PATH):
    print(f"ERROR: {DUMP_PATH} must be a file that exists.")

if os.path.exists(BASE_DOWNLOAD_PATH):
    if os.path.isdir(BASE_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {BASE_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {BASE_DOWNLOAD_PATH} is not a directory.")

if os.path.exists(MOL_DOWNLOAD_PATH):
    if os.path.isdir(MOL_DOWNLOAD_PATH):
        print(f"INFO: using existing directory {MOL_DOWNLOAD_PATH}")
    else:
        print(f"ERROR: {MOL_DOWNLOAD_PATH} is not a directory.")

if os.path.exists(BASE_SDF_OUTPUT_PATH):
    if os.path.isdir(BASE_SDF_OUTPUT_PATH):
        print(f"INFO: using existing directory {BASE_SDF_OUTPUT_PATH}")
    else:
        print(f"ERROR: {BASE_SDF_OUTPUT_PATH} is not a directory.")
else:
    print(f"WARNING: creating directory {BASE_SDF_OUTPUT_PATH}")
    os.mkdir(BASE_SDF_OUTPUT_PATH)

mols_casper = []
mols_smiles = []
casper_13c = []
casper_1h = []
duplicates = 0

print(f"INFO: reading CASPER mols...")

with Chem.SDMolSupplier("casper/casper_mono.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "CASPER")
        mol.SetProp("Class", "mono")
        mols_casper.append(mol)

with Chem.SDMolSupplier("casper/casper_dimer.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "CASPER")
        mol.SetProp("Class", "di")
        mols_casper.append(mol)

with Chem.SDMolSupplier("casper/casper_trimer.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "CASPER")
        mol.SetProp("Class", "tri")
        mols_casper.append(mol)

mols_glyconmr = []

print(f"INFO: reading GlycoNMR mols...")

with Chem.SDMolSupplier("glyconmr/glyconmr_mono.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "GlycoNMR")
        mol.SetProp("Class", "mono")
        mols_glyconmr.append(mol)

with Chem.SDMolSupplier("glyconmr/glyconmr_dimer.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "GlycoNMR")
        mol.SetProp("Class", "di")
        mols_glyconmr.append(mol)

with Chem.SDMolSupplier("glyconmr/glyconmr_trimer.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "GlycoNMR")
        mol.SetProp("Class", "tri")
        mols_glyconmr.append(mol)

with Chem.SDMolSupplier("glyconmr/glyconmr_tetra.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "GlycoNMR")
        mol.SetProp("Class", "tetra")
        mols_glyconmr.append(mol)

with Chem.SDMolSupplier("glyconmr/glyconmr_oligomer.sdf") as suppl:
    for mol in suppl:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)
        mol.SetProp("Source", "GlycoNMR")
        mol.SetProp("Class", "oligo")
        mols_glyconmr.append(mol)

print(f"INFO: loading CSDB dump file {DUMP_PATH}...")
dump = open(DUMP_PATH)
dump_lines = dump.readlines()
dump_data = sweetener.load_dump(dump_lines)
del(dump_lines)
dump.close()

mols_csdb = []

num_13c_missing = 0
num_1h_missing = 0
num_1h_loose = 0
num_illegal_atom = 0
num_illegal_solvent = 0
total = 0
total_kept = 0

for entry in os.scandir(MOL_DOWNLOAD_PATH):
    if entry.is_file():
        id = entry.name[:entry.name.find("-")]
        c13 = {}
        h1 = {}
        illegal = None
        solvent = ""
        for record in dump_data:
            if record.get('CSDB_ID') == id:
                total += 1
                structure: str = record.get('STRUCTURE')
                solvent: str = record.get('SOLVENT')
                if not (solvent == "D2O" or solvent.startswith("D2O; pH")):
                    illegal = "solvent"
                    break
                c13 = parse_csdb_line(record.get('CNMR SPECTRUM'))
                h1 = parse_csdb_line(record.get('HNMR SPECTRUM'))
                break
        else:
            print(f"ERROR: couldn't find {id} in CSDB dump file - aborting!")
            sys.exit()

        if illegal == "solvent":
            print(f"INFO: skipping {id} - illegal solvent...")
            num_illegal_solvent += 1
            continue

        mol = Chem.MolFromMolFile(entry.path)
        editable = Chem.EditableMol(mol)

        atoms_to_veto = set()
        numCarbons = 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_veto.add(atom.GetIdx())
                continue
            if not atom.GetAtomicNum() in (1, 6, 7, 8, 9, 15, 16, 17, 35):
                illegal = "atom"
                break
            if atom.GetAtomicNum() == 6:
                numCarbons += 1

        if illegal == "atom":
            print(f"INFO: skipping {id} - illegal atom...")
            num_illegal_atom += 1
            continue

        for idx in sorted(atoms_to_veto, reverse=True):
            editable.RemoveAtom(idx)

        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mol.RemoveAllConformers()

        nmr_mapping = mol.GetProp("_MolFileComments")[9:]

        res_maps = parse_mapping_line(nmr_mapping, atoms_to_veto)
        c13_spec, h1_spec = build_spectra(c13, h1, res_maps)

        if ((len(c13_spec) + 1) / numCarbons) < 0.8:
            print(f"INFO: skipping {id} - too many missing 13C peaks...")
            num_13c_missing += 1
            continue

        if ((len(h1_spec) + 1) / numCarbons) < 0.8:
            print(f"INFO: skipping {id} - too many missing 1H peaks...")
            num_1h_missing += 1
            continue

        tooHighVariance = False
        for peak in h1_spec.values():
            if peak[1] > 0.1:
                tooHighVariance = True
                break
        if tooHighVariance:
            print(f"INFO: skipping {id} - unacceptable 1H peak variance...")
            num_1h_loose += 1
            continue

        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in mols_smiles:
            print(f"INFO: skipping mol - duplicate")
            duplicates += 1
            continue
        mols_smiles.append(smiles)
        mol.SetProp("SMILES", smiles)

        mol.SetProp("13C Spectrum", str(c13_spec))
        mol.SetProp("1H Spectrum", str(h1_spec))

        structure = mol.GetProp("_Name")[14:]
        mol.SetProp("CSDB Linear", structure)
        mol.SetProp("_Name", entry.name[:entry.name.find('.')])
        mol.SetProp("Source", "CSDB")

        linkage_re = r"(\(.\-.\))|(\(.\-.\-.\))" #r"\(\d-\d\)"
        num_linkages = len(re.findall(linkage_re, structure))
        if num_linkages == 0:
            mol.SetProp("Class", "mono")
        elif num_linkages == 1:
            mol.SetProp("Class", "di")
        elif num_linkages == 2:
            mol.SetProp("Class", "tri")
        elif num_linkages == 3:
            mol.SetProp("Class", "tetra")
        elif num_linkages > 3 and num_linkages <= 10:
            mol.SetProp("Class", "oligo")
        elif num_linkages > 10:
            mol.SetProp("Class", "poly")
        else:
            print(f"ERROR: illegal number of linkages - aborting!")
            sys.exit()
        
        mols_csdb.append(mol)
        total_kept += 1

print(f"INFO: kept {total_kept} / {total} CSDB mols.")
print(f"INFO: {num_13c_missing} had too many missing 13C peaks, {num_1h_missing} had too many missing 1H peaks, {num_1h_loose} had too much 1H peak variance, {num_illegal_atom} had illegal atoms, {num_illegal_solvent} had an illegal solvent.")
print(f"INFO: skipped {duplicates} duplicates.")

print(f"INFO: writing SDF...")

mols_combined = mols_csdb + mols_casper + mols_glyconmr

random.seed(42)
random.shuffle(mols_combined)

with Chem.SDWriter(f"{BASE_SDF_OUTPUT_PATH}/combined.sdf") as w:
    for mol in mols_combined:
        w.write(mol)

print(f"INFO: wrote {len(mols_combined)} mols to SDF file.")
