# ChemScope

This repository contains the code used in the study [Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques](https://doi.org/10.1016/j.chemolab.2026.105659).
  
The scripts implement processing of literature data, physicochemical descriptor calculation, and chemometric analyses used to compare ligand chemical space across experimental techniques.

## Data Availability
The data used in this study is available at [Dataset](https://huggingface.co/datasets/erdemonal/ChemScope-Dataset)

## Requirements
Python ≥ 3.8  
Dependencies are listed in `requirements.txt`.

## Setup
```bash
git clone https://github.com/erdemonal/ChemScope.git
cd ChemScope
pip install -r requirements.txt
```

## Reproducing the Analysis

### Fetch Resources
Download essential chemical property datasets from OSF (Required as `data/raw` is not version controlled).
```bash
python fetch_resources.py
```

### Literature Mining
Mine Europe PMC for new protein-ligand associations. 
Define your search queries in `queries.txt` (format: `Name, "Search Query"`).
Example:
```text
ITC, "isothermal titration calorimetry" AND ("protein-ligand binding" OR "binding affinity")
```
Run the miner:
```bash
python literature_mining.py
```

### Data Processing
```bash
python data_processing.py -i data/interim -t folder
```

### Chemometric Analysis
```bash
python chemometrics_analysis.py
```

### Static Visualization
```bash
python static_visualization.py
```

### Interactive Visualization
```bash
python interactive_visualization.py -i data/processed
```

## Citation
If you use this code in your research, please cite the following paper:

**Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques**  
Erdem Önal, Zeynep Kalaycıoğlu  
*Chemometrics and Intelligent Laboratory Systems*, 2026, 105659  
[DOI: 10.1016/j.chemolab.2026.105659](https://doi.org/10.1016/j.chemolab.2026.105659)

### BibTeX
```bibtex
@article{ONAL2026105659,
title = {Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques},
journal = {Chemometrics and Intelligent Laboratory Systems},
pages = {105659},
year = {2026},
issn = {0169-7439},
doi = {https://doi.org/10.1016/j.chemolab.2026.105659},
url = {https://www.sciencedirect.com/science/article/pii/S0169743926000328},
author = {Erdem Önal and Zeynep Kalaycıoğlu},
keywords = {Affinity, bibliometrics, drug, visualization}
```
