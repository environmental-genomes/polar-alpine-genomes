# Code and Reproducibility Guide

## Polar and alpine environments deeply shaped microbial genomes

This repository contains analysis scripts associated with the manuscript **"Polar and alpine environments deeply shaped microbial genomes"**. The code supports sequence/genome preprocessing, protein-structure feature calculation, surface amino-acid statistics, and machine-learning classification of polar and alpine microbial genomes.

This README provides system requirements, installation notes, input data requirements, command-line examples, expected outputs, and figure-level workflow mapping.

## Repository Scope

The scripts are grouped into four main functional blocks:

1. **Genome and FASTA preprocessing**FASTA header normalization, FASTA splitting, assembly sequence cleaning, AAI-based genome de-duplication, and assembly-to-taxonomy replacement in phylogenetic trees.
2. **Protein structure feature calculation**Protein length, pLDDT, hydrogen bonds, disulfide bonds, isoelectric point, GOAP, KORP, surface amino-acid composition, and surface amino-acid property statistics.
3. **Surface amino-acid statistical testing**Welch t-tests comparing NPA and PAc groups based on summarized surface residue features.
4. **Machine-learning classification**
   Logistic Regression, SVM, Random Forest, MLP, model comparison, ROC curves, confusion matrices, and Integrated Gradients feature attribution.

## System Requirements

### Operating system

The scripts were originally written for a Linux environment. Recommended:

```text
Linux x86_64 or HPC cluster
Python 3.8-3.10
Perl 5
```

### Python dependencies

```text
biopython
numpy
pandas
scipy
scikit-learn
tensorflow
xgboost
matplotlib
seaborn
tqdm
mdtraj
```

### Example installation:

```bash
conda create -n pa-genomes python=3.10
conda activate pa-genomes
pip install biopython numpy pandas scipy scikit-learn tensorflow xgboost matplotlib seaborn tqdm mdtraj
```

### Perl dependencies

The Perl scripts use standard modules:

```text
Getopt::Long
File::Basename
```

### External software

Some structure feature scripts require third-party command-line tools:

```text
IPC      for isoelectric point calculation
Fast_GOAP for GOAP structural scoring
KORP / korpe for KORP structural scoring
```

The paths to these tools are hard-coded in `base_func.py` and must be modified before running on a new system.

## Required Input Data

### 1. FASTA or assembly files

Used by preprocessing scripts:

```text
*.fas
*.fasta
*.faa
assembly.fasta
```

### 2. AAI table

Used by genome de-duplication:

```text
aai.tsv
```

The default AAI column is column 6.

### 3. Assembly-to-taxonomy mapping and phylogenetic tree

Used by tree-label replacement:

```text
assembly2tax.txt
tree.nwk
```

### 4. Protein structure files

Structure feature scripts expect PDB files grouped by class:

```text
structure/
  NPA/
    protein_1.pdb
  PAc/
    protein_2.pdb
  PAd/
    protein_3.pdb
```

### 5. Machine-learning feature matrix

Machine-learning scripts generally expect a matrix with:

```text
rows    = features, genes, pathways, or categories
columns = genomes or samples
```

Example:

```text
feature_id,sample_1,sample_2,sample_3
gene001,0,1,2
gene002,3,0,5
```

### 6. Group label file

The group file should be tab-delimited and have no header:

```text
sample_1    NPA
sample_2    PAc
sample_3    PAd
```

## Configuration Before Running

Most Python scripts contain hard-coded paths from the original computing environment, for example:

```python
dir_file_path = '/public/home/lzzheng/zgl/project/microbio/analysis2/structure'
output = '/public/home/lzzheng/zgl/project/microbio/analysis2/statistic/length'
gene_file = r'\data\PA_transposition.csv'
group_file = r'\data\group.txt'
```

Before running, replace these with local paths.

External tool paths in `base_func.py` must also be edited:

```python
Fast_GOAP path
KORP korpe executable path
KORP score file path
IPC ipc.py path
```

## Quick Start

### FASTA preprocessing

Add file-name prefixes to FASTA headers:

```bash
perl addFileName.pl genome1.faa genome2.faa > all.rename.faa
```

Split a multi-FASTA file:

```bash
perl splitFas.pl --fasta all.rename.faa --mode 0
```

Clean an assembly FASTA:

```bash
python changeAssemblyContigLocusATGCN_v3.py assembly.fasta > assembly.clean.fasta
```

Remove near-duplicate genomes based on AAI:

```bash
perl de-duplication.pl --file aai.tsv --cutoff 99.9 --aai 6 > genome_deduplication.txt
```

Replace assembly IDs in a Newick tree:

```bash
perl assembly2taxonomy.pl assembly2tax.txt tree.nwk > tree.taxonomy.nwk
```

### Protein structure feature calculation

After editing the input and output paths in each script:

```bash
python cal_length.py
python cal_pLDDT.py
python cal_hydrogen_bonds.py
python cal_disulfide_bonds.py
python cal_PI.py
python cal_GOAP.py
python cal_korp.py
python cal_surface_aa.py
python cal_surface_aa_property.py
```

Calculate GOAP normalized by protein length:

```bash
python cal_per_length.py
```

### Surface amino-acid statistical testing

After generating the required summary JSON files:

```bash
python cls_surface_aa_p_value.py
python cls_surface_aa_property_p_value.py
```

### Machine-learning classification

Run individual models:

```bash
python model_LR.py
python model_SVM.py
python model_RF.py
python model_DeepLearning_MLP.py
```

Run model comparison:

```bash
python all_model_compare.py
```

Run MLP feature attribution:

```bash
python IG_explain_MLP.py
```

## Expected Outputs

### Structure feature scripts

Typical output structure:

```text
statistic/<feature>/
  NPA/
    protein_1.tsv
  PAc/
    protein_2.tsv
  PAd/
    protein_3.tsv
```

Each `.tsv` usually contains:

```text
protein_name    feature_value
```

Surface amino-acid scripts also write `.json` files for each protein.

### Statistical testing scripts

```text
proportions.csv
aa_surface_property.csv
```

### Machine-learning scripts

Possible outputs include:

```text
classification_report.csv
confusion_matrix.png or .pdf
roc_curve.png or .pdf
training_history.pdf
gene_importance_one_vs_rest_rf.txt
gene_importance_one_vs_rest_rf.pdf
all-model_metrics_summary_extended.csv
all_models_ROC_one_vs_rest.pdf
macro_average_ROC_comparison.pdf
IG_feature_importance_<class>.pdf
IG_feature_importance_heatmap.pdf
```

## Figure-Level Workflow Mapping

### Figure 1: Dataset and phylogeny

Relevant scripts:

```text
addFileName.pl
splitFas.pl
changeAssemblyContigLocusATGCN_v3.py
de-duplication.pl
assembly2taxonomy.pl
```

Purpose:

```text
FASTA normalization
assembly cleaning
genome de-duplication
tree label replacement
```

### Figure 4: Proteomic and structure features

Relevant scripts:

```text
base_func.py
cal_length.py
cal_pLDDT.py
cal_hydrogen_bonds.py
cal_disulfide_bonds.py
cal_PI.py
cal_GOAP.py
cal_korp.py
cal_per_length.py
cal_surface_aa.py
cal_surface_aa_property.py
cls_surface_aa_p_value.py
cls_surface_aa_property_p_value.py
```

Purpose:

```text
protein length
pLDDT
hydrogen bonds
disulfide bonds
isoelectric point
GOAP and KORP structural scores
surface amino-acid composition
surface amino-acid property statistics
NPA vs PAc statistical testing
```

### Figure 5: MLP and classifier models

Relevant scripts:

```text
model_LR.py
model_SVM.py
model_RF.py
model_DeepLearning_MLP.py
all_model_compare.py
IG_explain_MLP.py
```

Purpose:

```text
train and evaluate machine-learning classifiers
compare LR, SVM, RF, and MLP
generate ROC curves and confusion matrices
perform Integrated Gradients feature attribution
```

## Script Catalogue

| Script                                   | Purpose                                            | Main inputs                  | Main outputs                            |
| ---------------------------------------- | -------------------------------------------------- | ---------------------------- | --------------------------------------- |
| `addFileName.pl`                       | Add file-name prefix to FASTA headers              | FASTA files                  | Renamed FASTA on stdout                 |
| `splitFas.pl`                          | Split multi-FASTA into individual or grouped files | FASTA file                   | `.fas` files                          |
| `changeAssemblyContigLocusATGCN_v3.py` | Clean assembly FASTA and rename contigs            | assembly FASTA               | cleaned FASTA on stdout                 |
| `de-duplication.pl`                    | AAI-based genome de-duplication                    | AAI table                    | reserved and duplicated genome lists    |
| `assembly2taxonomy.pl`                 | Replace assembly IDs in Newick tree                | mapping table, tree          | taxonomy-labelled tree                  |
| `base_func.py`                         | Shared functions for PDB feature scripts           | PDB files, external tools    | helper functions                        |
| `cal_length.py`                        | Protein length                                     | PDB directory                | length`.tsv` files                    |
| `cal_pLDDT.py`                         | Mean pLDDT from B-factor                           | PDB directory                | pLDDT`.tsv` files                     |
| `cal_hydrogen_bonds.py`                | Approximate hydrogen bonds per length              | PDB directory                | hydrogen bond`.tsv` files             |
| `cal_disulfide_bonds.py`               | Disulfide bond count                               | PDB directory                | disulfide bond`.tsv` files            |
| `cal_PI.py`                            | Isoelectric point through IPC                      | PDB directory                | pI`.tsv` files                        |
| `cal_GOAP.py`                          | GOAP structural score                              | PDB directory                | GOAP`.tsv` files                      |
| `cal_korp.py`                          | KORP structural score                              | PDB directory                | KORP`.tsv` files                      |
| `cal_per_length.py`                    | GOAP per protein length                            | GOAP and length outputs      | normalized score`.tsv` files          |
| `cal_surface_aa.py`                    | Surface amino-acid fractions                       | PDB directory                | `.tsv` and `.json` files            |
| `cal_surface_aa_property.py`           | Surface residue property fractions                 | PDB directory                | `.tsv` and `.json` files            |
| `cls_surface_aa_p_value.py`            | Welch t-test for surface amino acids               | summary JSON                 | `proportions.csv`                     |
| `cls_surface_aa_property_p_value.py`   | Welch t-test for surface residue properties        | summary JSON                 | `aa_surface_property.csv`             |
| `model_LR.py`                          | Logistic Regression classifier                     | feature matrix, group labels | reports, ROC, feature coefficients      |
| `model_SVM.py`                         | SVM classifier                                     | feature matrix, group labels | result PDF                              |
| `model_RF.py`                          | Random Forest classifier                           | feature matrix, group labels | result PDF                              |
| `model_DeepLearning_MLP.py`            | MLP classifier and feature-gene extraction         | feature matrix, group labels | ROC, confusion matrix, training history |
| `all_model_compare.py`                 | Unified model comparison                           | feature matrix, group labels | model metrics and ROC PDFs              |
| `IG_explain_MLP.py`                    | Integrated Gradients explanation for MLP           | feature matrix, group labels | IG plots and heatmap                    |

## Reproducibility Notes

1. The scripts contain hard-coded paths from the original computing environment. These paths must be replaced before running.
2. The scripts  are not a single end-to-end pipeline. Some intermediate files, such as feature matrices and summary JSON files, must be generated by external workflows or prepared before running downstream scripts.
3. Several structure-scoring scripts depend on external software that is not bundled with this repository.

## Data Availability

Raw genomes, PDB files, derived feature matrices, and external database outputs are not bundled with these scripts unless explicitly included elsewhere in the repository. Users should prepare the required input files according to the formats described above.

## Code Availability

This repository provides analysis scripts and documentation for reproducing key computational steps described in the manuscript.
