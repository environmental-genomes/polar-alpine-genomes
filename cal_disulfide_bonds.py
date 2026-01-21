#!/public/home/guoliangzhu/miniconda3/envs/DMS/bin/python3.9
import os 
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import os
from Bio.PDB import PDBParser
import numpy as np

def run_command(command):
    import pty
    import subprocess
    import os
    import sys
    import select
    import errno

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
        name = os.path.basename(fpdb)
        protein_name = name.split(f'./')[0]

        result = func(fpdb)

        with open(fout, 'w') as f:
            f.write(f'{protein_name}\t{result}\n')

        return result

    return wrapper

    
@handle_output_decorator
def calculate_disulfide_bonds(pdb_file, threshold=2.2):
    """
    Calculate the number of disulfide bonds in a PDB file.
    
    Parameters:
        pdb_file (str): Path to the PDB file.
        threshold (float): Distance threshold for disulfide bond (default: 2.2 Å).
    
    Returns:
        int: Number of disulfide bonds.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    cysteines = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue is a cysteine
                if residue.get_resname() == "CYS" and "SG" in residue:
                    cysteines.append(residue["SG"].get_coord())

    # Calculate distances and count disulfide bonds
    cysteine_coords = np.array(cysteines)
    disulfide_bonds = 0
    for i in range(len(cysteine_coords)):
        for j in range(i + 1, len(cysteine_coords)):
            distance = np.linalg.norm(cysteine_coords[i] - cysteine_coords[j])
            if distance <= threshold:
                disulfide_bonds += 1

    return disulfide_bonds
    

def process_protein(protein_pdb_file, fout):
    calculate_disulfide_bonds(protein_pdb_file, fout)

def main():
    dir_file_path = f'/public/home/lzzheng/zgl/project/microbio/analysis2/structure'
    output = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/disulfide_bonds'
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

if __name__ == "__main__":
    main()

