import pty
import subprocess
import os
import sys
import select
import errno
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import wraps
from Bio.PDB import PDBParser
import shutil
import numpy as np
from Bio import PDB

def run_command(command):


    master, slave = pty.openpty()
    process = subprocess.Popen(command, shell=True, stdout=slave, stderr=slave, stdin=slave, universal_newlines=True)
    os.close(slave)  # Close slave descriptor in parent process

    output = []

    try:
        while process.poll() is None:
            if master < 0:
                break  # Check if master descriptor is valid

            rlist, _, _ = select.select([master], [], [], 0.1)
            if rlist:
                try:
                    data = os.read(master, 1024).decode('utf-8', errors='ignore')
                    if data:
                        output.append(data)
                        #sys.stdout.write(data)
                        #sys.stdout.flush()
                except OSError as e:
                    if e.errno == errno.EIO:  # Handle IOError
                        break
    finally:
        if master >= 0:
            os.close(master)

    return ''.join(output)


def handle_output_decorator(func):
    
    def wrapper(fpdb, fout):
        
        if fout and os.path.exists(fout) and os.path.getsize(fout) > 0:
            return
        
        
        name = os.path.basename(fpdb)
        protein_name = name.split(f'./')[0]

        result = func(fpdb)

        with open(fout, 'w') as f:
            f.write(f'{protein_name}\t{result}\n')

        return result

    return wrapper




def main(dir_file_path,output,process_protein):

    os.makedirs(output, exist_ok=True)

    tasks = []
    for protein_cls_name in os.listdir(dir_file_path):
        foutput = os.path.join(output, protein_cls_name)
        os.makedirs(foutput, exist_ok=True)
    
        protein_cls_name_path = os.path.join(dir_file_path, protein_cls_name)
        for protein_pdb in os.listdir(protein_cls_name_path):
            protein_pdb_file = os.path.join(protein_cls_name_path, protein_pdb)
            protein_name = protein_pdb.split('.')[0]
            fout = os.path.join(foutput, f'{protein_name}.tsv')
            tasks.append((protein_pdb_file, fout))

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_protein, task[0], task[1]): task for task in tasks}

        # Display progress bar using tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Proteins"):
            try:
                future.result()
            except Exception as e:
                protein_pdb_file, fout = futures[future]
                print(f"Error processing {protein_pdb_file}: {e}")
                
                

def calculate_pLDDT_average(fpdb,fout):
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', fpdb)
    
    b_factor_sum = 0.0
    atom_count = 0
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b_factor_sum += atom.get_bfactor()
                    atom_count += 1
    
    if atom_count == 0:
        return None  
    
    return b_factor_sum / atom_count




three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
standard_residues = set(three_to_one.keys())
def extract_amino_acid_sequence_single_chain(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                if residue.get_resname() in standard_residues:  # 只提取标准氨基酸
                    resname = residue.get_resname()
                    if resname in three_to_one:  # 转换为单字母代码
                        residues.append(three_to_one[resname])
            return "".join(residues)
        

def cal_length(fpdb):
    sequence = extract_amino_acid_sequence_single_chain(fpdb)
    return len(sequence)



def calculate_residue_hydrogen_bonds(pdb_file, distance_cutoff=3.5):
    
    """
    Calculate hydrogen bonds between protein residues in a PDB file.

    Parameters:
        pdb_file (str): Path to the PDB file.
        distance_cutoff (float): Maximum distance (in Å) for hydrogen bonds (default: 3.5 Å).

    Returns:
        int: Number of hydrogen bonds between residues.
    """
    
    
    length = cal_length(pdb_file)
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    donor_atoms = []
    acceptor_atoms = []

    # Extract donor (N, H) and acceptor (O) atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("N"):
                    donor_atoms.append(residue["N"].get_coord())  # Add N as donor
                if residue.has_id("O"):
                    acceptor_atoms.append(residue["O"].get_coord())  # Add O as acceptor

    # Calculate distances between all donors and acceptors
    donor_coords = np.array(donor_atoms)
    acceptor_coords = np.array(acceptor_atoms)

    hbond_count = 0
    for donor in donor_coords:
        for acceptor in acceptor_coords:
            distance = np.linalg.norm(donor - acceptor)
            if distance <= distance_cutoff:
                hbond_count += 1

    return hbond_count /  length 

def cal_korpe(fpdb):
    
    name = os.path.basename(fpdb)
    destination_file = os.path.join(os.getcwd(), name)

    shutil.copy(fpdb, destination_file)
  
    cmd = f'/public/home/guoliangzhu/bioapp/korp/Korp6Dv1/bin/korpe {destination_file} --score_file /public/home/guoliangzhu/bioapp/korp/Korp6Dv1/korp6Dv1.bin --only_score'
    output = run_command(cmd)
    os.remove(destination_file)
    
    korpe_score = output.split('\n')[-1]
        
    return  korpe_score


def pdb_to_fasta(pdb_file, output_fasta):
    """
    Convert a PDB file to a FASTA file.
    
    Parameters:
        pdb_file (str): Path to the input PDB file.
        output_fasta (str): Path to the output FASTA file.
    """
    # Create a parser to read the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Initialize a list to hold the amino acid sequence
    sequence = []
    
    # Loop through each model, chain, and residue to get the sequence
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only consider standard amino acids
                if PDB.is_aa(residue):
                    sequence.append(residue.get_resname())
    
    # Convert the 3-letter codes to 1-letter codes
    aa_1_letter = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    
    sequence = [aa_1_letter[res] for res in sequence]
    
    # Write the sequence to the output FASTA file
    with open(output_fasta, "w") as f:
        f.write(">protein_sequence\n")
        f.write("".join(sequence) + "\n")
        
    return output_fasta


def cal_PI(fpdb):
    
    protein_name = os.path.basename(fpdb).strip(f'.pdb')
    output_fasta = protein_name + f'.fasta'
    pdb_to_fasta(fpdb, output_fasta)
    cmd = f'/public/home/guoliangzhu/miniconda3/envs/DMS/bin/python3.9 /public/home/guoliangzhu/bioapp/IPC/ipc-2.0.1/scripts/ipc2_lib/ipc.py "{output_fasta}"'
    run_command(cmd)
    
    result_file = output_fasta + f'.pI.txt'
    
    start = False
    with open(result_file,'r') as f:
        for line in f.readlines():
            if line.startswith('>'):
                start = True
                continue
            if start:
                score = float(line.split()[-1])
                break
                
    os.remove(output_fasta)
    os.remove(result_file)
    return score

def cal_goap(fpdb):
    
    cmd = f'/public/home/lzzheng/apps/zSaprod/zSaprod/bin/Fast_GOAP -i "{fpdb}"'
    output = run_command(cmd)
    goap_score = output.split()[-1]
    
    try:
        goap_score = float(goap_score)
    except:
        print(goap_score)
    
    
    return goap_score


