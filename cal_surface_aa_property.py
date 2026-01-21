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
    "polar", "nonpolar", "aromatic"
    ]
    
    name = os.path.basename(fpdb)
    protein_name = name.split(f'./')[0]
    result = count_surface_residues_with_types(fpdb)
    
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


def count_surface_residues_with_types(pdb_file, cutoff=0.5):
    # 氨基酸分类
    # 极性氨基酸，包括带电荷和不带电荷的
    # 非极性氨基酸
    nonpolar_residues = {"GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "MET", "PHE", "TRP"}
    polar_residues = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "LYS", "ARG", "HIS", "ASP", "GLU"}
    aromatic_residues = {"PHE", "TYR", "TRP"}

    
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

    # 统计表面极性、非极性、芳香氨基酸的数量
    polar_count = sum(residue_counts[res] for res in polar_residues if res in residue_counts)
    nonpolar_count = sum(residue_counts[res] for res in nonpolar_residues if res in residue_counts)
    aromatic_count = sum(residue_counts[res] for res in aromatic_residues if res in residue_counts)

    total_surface_residues = len(surface_residues)
    residue_fractions = {
        "polar": polar_count / total_surface_residues,
        "nonpolar": nonpolar_count / total_surface_residues,
        "aromatic": aromatic_count / total_surface_residues,
    }

    return residue_fractions



def process_protein(protein_pdb_file,fout):
    handle_results(protein_pdb_file,fout)
    
    


if __name__ == "__main__":
    
    dir_file_path =  f'/public/home/lzzheng/zgl/project/microbio/analysis2/structure'
    output = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/surface_aa_property'    
    main(dir_file_path,output,process_protein)

