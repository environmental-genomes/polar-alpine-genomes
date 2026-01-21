#!/data_test/lzzheng/apps/pythonProject1/envs/DMS/bin/python
from base_func import handle_output_decorator,main,cal_PI

@handle_output_decorator
def cal_func(fpdb):
    result = cal_PI(fpdb)
    return result
    

def process_protein(protein_pdb_file,fout):
    cal_func(protein_pdb_file,fout)


if __name__ == "__main__":
    
    dir_file_path = f'/public/home/lzzheng/zgl/project/microbio/analysis2/structure'
    output = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/PI'
    main(dir_file_path,output,process_protein)


