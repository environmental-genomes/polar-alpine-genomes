#!/public/home/guoliangzhu/miniconda3/envs/DMS/bin/python3.9

import os
import json
import numpy as np
import matplotlib.pyplot as plt


file_json = f'/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/summary_aa_surface_property.json'


with open(file_json, 'r') as f:
    data = json.load(f)


from scipy.stats import ttest_ind

# 示例数据
# data = {
#     "protein1": {
#         "NPA": [{"MET": 1, "HIS": 5, "LYS": 3}, {"MET": 3, "HIS": 5, "LYS": 3}],
#         "PAc": [{"MET": 3, "HIS": 4, "LYS": 2}, {"MET": 1, "HIS": 5, "LYS": 3}, {"MET": 1, "HIS": 5, "LYS": 3}],
#     },
#     "protein2": {
#         "NPA": [{"MET": 2, "HIS": 6, "LYS": 1}, {"MET": 1, "HIS": 5, "LYS": 3}],
#         "PAc": [{"MET": 4, "HIS": 7, "LYS": 3}, {"MET": 1, "HIS": 5, "LYS": 3}, {"MET": 1, "HIS": 5, "LYS": 3}, {"MET": 1, "HIS": 5, "LYS": 4}],
#     },
# }

# 计算 p 值函数
def calculate_p_values(data):
    results = {}
    all_amino_acids = set()

    for protein_name, groups in data.items():
        npa_samples = groups.get("NPA", [])
        pac_samples = groups.get("PAc", [])
        results[protein_name] = {}

        # 获取所有氨基酸的集合
        for sample in npa_samples + pac_samples:
            all_amino_acids.update(sample.keys())

        for aa in all_amino_acids:
            # 提取 NPA 和 PAc 中当前氨基酸的计数值
            npa_values = [sample.get(aa, 0) for sample in npa_samples]
            pac_values = [sample.get(aa, 0) for sample in pac_samples]

            # 计算 t 检验
            _, p_value = ttest_ind(npa_values, pac_values, equal_var=False)

            # 保存结果
            results[protein_name][aa] = {"p_value": p_value, "npa_mean": np.mean(npa_values), "pac_mean": np.mean(pac_values)}

    return results, all_amino_acids

# 计算 p 值并提取所有氨基酸
p_values, all_amino_acids = calculate_p_values(data)

# 统计 NPA < PAc 和 NPA > PAc 且 p 值小于 0.05 的占比
def calculate_proportions(p_values, all_amino_acids, threshold=0.05):
    # 初始化字典
    proportions_less = {aa: 0 for aa in all_amino_acids}
    proportions_greater = {aa: 0 for aa in all_amino_acids}
    proportions_p_value_less_05_less = {aa: 0 for aa in all_amino_acids}
    proportions_p_value_less_05_greater = {aa: 0 for aa in all_amino_acids}
    proportions_p_value_greater_05_less = {aa: 0 for aa in all_amino_acids}
    proportions_p_value_greater_05_greater = {aa: 0 for aa in all_amino_acids}

    total_proteins = len(p_values)

    for aa in all_amino_acids:
        count_less = 0
        count_greater = 0
        count_p_value_less_05_less = 0
        count_p_value_less_05_greater = 0
        count_p_value_greater_05_less = 0
        count_p_value_greater_05_greater = 0

        for protein, aa_values in p_values.items():
            if aa in aa_values:
                p_value = aa_values[aa]["p_value"]
                npa_mean = aa_values[aa]["npa_mean"]
                pac_mean = aa_values[aa]["pac_mean"]

                # p_value < threshold
                if p_value < threshold:
                    count_less += 1
                    if npa_mean < pac_mean:
                        count_p_value_less_05_less += 1
                    elif npa_mean > pac_mean:
                        count_p_value_less_05_greater += 1

                # p_value > 0.5
                elif p_value >= threshold:
                    count_greater += 1
                    if npa_mean < pac_mean:
                        count_p_value_greater_05_less += 1
                    elif npa_mean > pac_mean:
                        count_p_value_greater_05_greater += 1

        # 计算比例
        proportions_less[aa] = count_less / total_proteins
        proportions_greater[aa] = count_greater / total_proteins
        proportions_p_value_less_05_less[aa] = count_p_value_less_05_less / total_proteins
        proportions_p_value_less_05_greater[aa] = count_p_value_less_05_greater / total_proteins
        proportions_p_value_greater_05_less[aa] = count_p_value_greater_05_less / total_proteins
        proportions_p_value_greater_05_greater[aa] = count_p_value_greater_05_greater / total_proteins

    return {
        "proportions_less": proportions_less,
        "proportions_greater": proportions_greater,
        "proportions_p_value_less_05_less": proportions_p_value_less_05_less,
        "proportions_p_value_less_05_greater": proportions_p_value_less_05_greater,
        "proportions_p_value_greater_05_less": proportions_p_value_greater_05_less,
        "proportions_p_value_greater_05_greater": proportions_p_value_greater_05_greater,
    }


import csv

def save_proportions_to_csv(result, output_file):
    # 提取所有氨基酸和对应比例
    amino_acids = list(result["proportions_less"].keys())
    headers = [
        "aa",  # 氨基酸列
        "proportions_less_0.05",
        "proportions_greater_0.05",
        "p_less_0.05_npa_less_pac", 
        "p_less_0.05_npa_greater_pac",
        "p_greater_0.05_npa_less_pac", 
        "p_greater_0.05_npa_greater_pac",

    ]

    # 组织每个氨基酸的行数据
    rows = []
    for aa in amino_acids:
        rows.append([
            aa,
            result["proportions_less"].get(aa, 0), 
            result["proportions_greater"].get(aa, 0),
            result["proportions_p_value_less_05_less"].get(aa, 0),
            result["proportions_p_value_less_05_greater"].get(aa, 0),
            result["proportions_p_value_greater_05_less"].get(aa, 0),
            result["proportions_p_value_greater_05_greater"].get(aa, 0),
        ])

    # 写入 CSV 文件
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Proportions saved to {output_file}")


# 调用示例
result = calculate_proportions(p_values, all_amino_acids, threshold=0.05)
save_proportions_to_csv(result, "aa_surface_property.csv")