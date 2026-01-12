# ChemScope

This repository contains the code used in the study  
“Text Mining-Based Profiling of Chemical Environments in Protein–Ligand Binding Assays Across Analytical Techniques.”

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