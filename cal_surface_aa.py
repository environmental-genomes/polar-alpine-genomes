#!/public/home/guoliangzhu/miniconda3/envs/DMS/bin/python3.9
from base_func import main
import mdtraj as md
from collections import Counter
import os
import json


    
def handle_results(fpdb, fout):
    
    if fout and os.path.exists(fout) and os.path.getsize(fout) > 0:
        return
    
    aa_list = [
    'MET', 'PHE', 'ASP', 'ILE', 'LYS', 'ARG', 'THR', 'GLU', 'GLY', 'GLN', 
    'VAL', 'SER', 'PRO', 'LEU', 'ASN', 'TYR', 'HIS', 'ALA', 'TRP', 'CYS'
    ]
    
    name = os.path.basename(fpdb)
    protein_name = name.split(f'./')[0]
    result = count_surface_residues(fpdb)
    
    dir_name = os.path.dirname(fout)
    fout_name = os.path.basename(fout).split(f'.')[0] + f'.json'
    fjson = os.path.join(dir_name,fout_name)
    with open(fjson,'w') as f:
        json.dump(result,f,indent=4) 
        
    line = protein_name + f'\t'
    for aa in aa_list:
        
        if aa not in result:
            result[aa] = 0
        line  += f'{result[aa]}' + f'\t'
    line = line[:-1] + f'\n'
            
            
    with open(fout, 'w') as f:
        
        f.write(line)
        
 
    return 


def count_surface_residues(pdb_file, cutoff=0.5):
    # 加载 PDB 文件
    traj = md.load(pdb_file)

    # 计算每个原子的表面积
    sasa = md.shrake_rupley(traj, mode='residue')
    sasa_per_residue = sasa[0]

    # 获取氨基酸信息并转换为列表
    residues = list(traj.topology.residues)  # 转换生成器为列表
    surface_residues = [res for res, area in zip(residues, sasa_per_residue) if area > cutoff]

    # 统计表面氨基酸的种类和数目
    surface_residue_names = [res.name for res in surface_residues]
    residue_counts = Counter(surface_residue_names)
    
    total_surface_residues = len(surface_residues)
    residue_fractions = {name: count / total_surface_residues for name, count in residue_counts.items()}
    

    return residue_fractions



def process_protein(protein_pdb_file,fout):
    handle_results(protein_pdb_file,fout)
    
    


if __name__ == "__main__":
    
    dir_file_path =  f'/public/home/lzzheng/zgl/project/microbio/analysis2/structure'
    output = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/surface_aa'    
    main(dir_file_path,output,process_protein)

