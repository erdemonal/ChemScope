# ChemScope

This repository contains the code used in the study "Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques."
  
The scripts implement processing of literature data, physicochemical descriptor calculation, and chemometric analyses used to compare ligand chemical space across experimental techniques.

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