# ChemScope

This repository contains the code used in the study [Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques](https://doi.org/10.1016/j.chemolab.2026.105659).

## Data Availability
The data used in this study is available at:
- Hugging Face: [ChemScope-Dataset](https://huggingface.co/datasets/erdemonal/ChemScope-Dataset) 
- GitHub: [Integrated_Physicochemical_Dataset.csv](data/processed/Integrated_Physicochemical_Dataset.csv)

## Setup
In order to run the project, you will need Python 3.8 or above. The required libraries listed in `requirements.txt` must be installed.

```bash
git clone https://github.com/erdemonal/ChemScope.git
cd ChemScope
pip install -r requirements.txt
```

## Reproducing the Analysis

### Fetch Resources
Download chemical property datasets from OSF (Required as `data/raw` is not version controlled).
```bash
python scripts/fetch_resources.py
```

### Literature Mining
Mine Europe PMC for protein-ligand associations. 
Define your search queries in `queries.txt` (format: `Name, "Search Query"`).
Example:
```text
ITC, "isothermal titration calorimetry" AND ("protein-ligand binding" OR "binding affinity")
```
Run the miner:
```bash
python scripts/literature_mining.py
```

### Data Processing
```bash
python scripts/data_processing.py -i data/interim -t folder
```

### Chemometric Analysis
```bash
python scripts/chemometrics_analysis.py
```

### Static Visualization
```bash
python scripts/static_visualization.py
```

### Interactive Visualization
```bash
python scripts/interactive_visualization.py -i data/processed
```

## Citation
If you use this code in your research, please cite the following paper:

Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques
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
