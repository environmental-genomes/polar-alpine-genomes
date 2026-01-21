#!/public/home/guoliangzhu/miniconda3/envs/DMS/bin/python3.9
import os


attr_file_data_file = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/goap'
length_file_file = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/length'
out_path = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/'
name = f'goap_per_length'


fout_path = os.path.join(out_path,name)
os.makedirs(fout_path,exist_ok=True)



pro_to_length = {}
for protein_cls_file in os.listdir(length_file_file):
    
    protein_dir = os.path.join(length_file_file,protein_cls_file)
    
    for protein_name in os.listdir(protein_dir):
        protein_data_file = os.path.join(protein_dir, protein_name)
        
        with open(protein_data_file, 'r') as f:
            pro_name, score = f.readline().strip('\n').split('\t')
            pro_name = pro_name.strip('.pdb')
            pro_to_length[pro_name] = int(score)
            

for protein_cls_file in os.listdir(attr_file_data_file):
    
    protein_dir = os.path.join(attr_file_data_file,protein_cls_file)
    
    pro_cls_out_file = os.path.join(fout_path,protein_cls_file)
    os.makedirs(pro_cls_out_file,exist_ok=True)
    
    
    for protein_name in os.listdir(protein_dir):
        protein_data_file = os.path.join(protein_dir, protein_name)
        
        protein_output_data_file = os.path.join(pro_cls_out_file, protein_name)
        
        with open(protein_data_file, 'r') as f,open(protein_output_data_file,'w') as f2:
            pro_name, score = f.readline().strip('\n').split('\t')
            pro_name = pro_name.strip('.pdb')
            try:
                score = float(score)
            except:
                print(protein_data_file)
                
            length = pro_to_length[pro_name]
            
            score_per_length = score / length
            
            f2.write(f'{pro_name}\t{score_per_length}\n')
            
            


            
        
    
    


