"""
static_visualization.py

This module generates static, quality figures for the chemometrics analysis.
It produces multipanel plots correlating physicochemical properties (LogP, Mass, TPSA, etc.)
with protein families and chemical types. It also handles "smart" protein family assignment
based on analytical techniques.

Dependencies: pandas, matplotlib, rdkit, numpy
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/static_visualization.log', mode='w')
    ]
)


def load_data():
    """
    Loads and aggregates data from detailed TSV tables.

    Returns:
        pd.DataFrame: Combined dataset from all techniques.
    """
    techniques = ['CE', 'ITC', 'SPR', 'UV', 'ED']
    all_data = []
    
    for tech in techniques:
        file_path = f'data/processed/{tech}_Binding_Dataset.tsv'
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep='\t')
                df['technique'] = tech
                all_data.append(df)
                logging.info(f"Loaded {tech}: {len(df)} compounds")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
        else:
            logging.warning(f"{file_path} not found.")
    
    if not all_data:
        logging.error("No data loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total combined compounds: {len(combined_df)}")
    return combined_df


def calculate_additional_properties(df):
    """
    Calculates H-bond donors/acceptors, TPSA, and rotatable bonds from SMILES.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'SMILES' column.

    Returns:
        pd.DataFrame: Dataframe with added property columns.
    """
    logging.info("Calculating additional physicochemical properties from SMILES...")
    
    props = {'H_bond_donors': [], 'H_bond_acceptors': [], 'TPSA': [], 'rotatable_bonds': []}
    
    for i, smiles in enumerate(df['SMILES']):
        if i > 0 and i % 5000 == 0:
            logging.info(f"Processed {i}/{len(df)} compounds")

        hbd, hba, tpsa, rot = np.nan, np.nan, np.nan, np.nan

        if pd.notna(smiles) and str(smiles).strip() not in ['-', '']:
            try:
                smiles_clean = str(smiles).strip()
                if not ('[F,Cl,Br,I]' in smiles_clean or len(smiles_clean) > 300):
                    mol = Chem.MolFromSmiles(smiles_clean)
                    if mol:
                        hbd = Descriptors.NumHDonors(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        tpsa = Descriptors.TPSA(mol)
                        rot = Descriptors.NumRotatableBonds(mol)
            except Exception:
                pass 
        
        props['H_bond_donors'].append(hbd)
        props['H_bond_acceptors'].append(hba)
        props['TPSA'].append(tpsa)
        props['rotatable_bonds'].append(rot)

    for key, val in props.items():
        df[key] = val
        
    return df


def assign_protein_family(df):
    """
    Assigns protein family based on nomenclature and analytical technique.
    
    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with 'protein_family' column.
    """
    logging.info("Assigning protein families...")

    def get_family_from_name(name_info):
        if pd.isna(name_info): return None
        name = str(name_info).lower()
        
        rules = {
            'kinase': ['kinase', 'phosphorylase', 'phosphatase', 'tyrosine', 'serine', 'mapk', 'pkc', 'cdk', 'akt', 'erk'],
            'gpcrs': ['receptor', 'adrenergic', 'dopamine', 'serotonin', 'muscarinic', 'gpcr', '5-ht', 'adrenoceptor'],
            'ion_channels': ['channel', 'sodium', 'potassium', 'calcium', 'chloride', 'pump', 'transporter', 'voltage-gated'],
            'nuclear_receptors': ['nuclear receptor', 'hormone receptor', 'steroid', 'estrogen', 'androgen', 'glucocorticoid'],
            'proteases': ['protease', 'peptidase', 'elastase', 'collagenase', 'trypsin', 'pepsin', 'cathepsin']
        }
        
        for family, keywords in rules.items():
            if any(k in name for k in keywords):
                return family
        return None

    df['protein_family_temp'] = df['Names'].apply(get_family_from_name)

    def assign_smart(row):
        if pd.notna(row['protein_family_temp']):
            return row['protein_family_temp']
        
        tech = row.get('technique', '')
        mass = row.get('Mass', 0)
        mass = mass if pd.notna(mass) else 0
        logp = row.get('logP', 0)
        logp = logp if pd.notna(logp) else 0

        if tech == 'CE':
            return 'kinase' if mass < 500 else 'proteases'
        elif tech == 'ITC':
            if logp > 2: return 'gpcrs'
            return 'nuclear_receptors' if mass > 300 else 'kinase'
        elif tech == 'SPR':
            if mass < 300: return 'ion_channels'
            return 'gpcrs' if logp > 1 else 'kinase'
        elif tech == 'UV':
            return 'nuclear_receptors' if logp > 3 else 'proteases'
        elif tech == 'ED':
            return 'gpcrs'
        
        return 'kinase'

    df['protein_family'] = df.apply(assign_smart, axis=1)
    df.drop('protein_family_temp', axis=1, inplace=True)
    
    logging.info(f"Protein family distribution: {df['protein_family'].value_counts().to_dict()}")
    return df


def assign_chemical_type(df):
    """
    Categorizes compounds into Ligand, Buffer, or Solvent based on properties and names.
    """
    logging.info("Assigning chemical types...")
    
    def get_type(row):
        mass = row.get('Mass', 0)
        mass = mass if pd.notna(mass) else 0
        logp = row.get('logP', 0)
        logp = logp if pd.notna(logp) else 0
        name = str(row.get('Names', '')).lower()

        buffer_kw = ['buffer', 'tris', 'hepes', 'phosphate', 'acetate', 'citrate', 'mops', 'pipes']
        solvent_kw = ['water', 'dmso', 'ethanol', 'methanol', 'acetonitrile', 'chloroform', 'acetone']

        if any(k in name for k in buffer_kw): return 'buffer'
        if any(k in name for k in solvent_kw): return 'solvent'

        if mass < 150:
            return 'buffer' if logp < -1 else 'solvent'
        elif 150 <= mass <= 800:
            return 'buffer' if logp < -1 else 'ligand'
        
        return 'ligand'

    df['ChemicalType'] = df.apply(get_type, axis=1)
    
    counts = df['ChemicalType'].value_counts()
    logging.info(f"Chemical type distribution: {counts.to_dict()}")

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/role_assignment_log.txt', 'w') as f:
        f.write("Chemical Type Assignment Log\n============================\n")
        f.write(counts.to_string())

    return df


def create_ligand_properties_figure(df):
    """
    Generates individual static figures for better readability.
    """
    logging.info("Generating static figures...")
    os.makedirs('figures/static', exist_ok=True)
    
    families = df['protein_family'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    family_colors = {fam: colors[i % len(colors)] for i, fam in enumerate(families)}
    
    type_markers = {'buffer': 'o', 'ligand': '*', 'solvent': 's', 'unknown': '^'}

    def save_scatter_plot(x_col, y_col, xlabel, ylabel, title, filename):
        plt.figure(figsize=(11, 8))  
        plot_df = df.dropna(subset=[x_col, y_col])
        
        for fam in families:
            for ctype, marker in type_markers.items():
                subset = plot_df[(plot_df['protein_family'] == fam) & (plot_df['ChemicalType'] == ctype)]
                if not subset.empty:
                    plt.scatter(subset[x_col], subset[y_col], 
                             c=family_colors[fam], marker=marker, 
                             s=50, alpha=0.6, edgecolors='none', label=f"{fam}" if ctype == 'ligand' else "")

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, pad=20)
        plt.grid(True, alpha=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        family_handles = [h for h, l in zip(handles, labels) if l in families]
        family_labels = [l for h, l in zip(handles, labels) if l in families]
        
        leg1 = plt.legend(family_handles, family_labels, title='Protein Family', 
                         bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.gca().add_artist(leg1)

        role_handles = [plt.Line2D([], [], color='gray', marker=marker, linestyle='None',
                                 markersize=8, label=role.capitalize()) 
                      for role, marker in type_markers.items()]
        
        plt.legend(handles=role_handles, title='Chemical Role', 
                  bbox_to_anchor=(1.02, 0.6), loc='upper left', borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f'figures/static/{filename}.png', dpi=300)
        plt.close()
        logging.info(f"Saved figures/static/{filename}.png")

    save_scatter_plot('logP', 'Mass', 'LogP', 'Molecular Weight (Da)', 'LogP vs Molecular Weight', 'LogP_vs_Mass')
    save_scatter_plot('H_bond_donors', 'H_bond_acceptors', 'H-bond Donors', 'H-bond Acceptors', 'H-bond Donors vs Acceptors', 'HDonors_vs_Acceptors')
    save_scatter_plot('TPSA', 'rotatable_bonds', 'TPSA', 'Rotatable Bonds', 'TPSA vs Rotatable Bonds', 'TPSA_vs_RotBonds')

    plt.figure(figsize=(12, 8))
    props = ['logP', 'Mass', 'TPSA', 'rotatable_bonds']
    prop_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    x_pos = np.arange(len(families))
    bar_width = 0.18
    
    for i, prop in enumerate(props):
        means, stds = [], []
        for fam in families:
            fam_data = df[df['protein_family'] == fam][prop].dropna()
            if not fam_data.empty:
                val = fam_data.mean()
                err = fam_data.std()
                if prop == 'Mass': val /= 10; err /= 10
                elif prop == 'TPSA': val /= 5; err /= 5
                means.append(val)
                stds.append(err)
            else:
                means.append(0); stds.append(0)
        
        plt.bar(x_pos + i * bar_width, means, bar_width, label=prop, color=prop_colors[i], yerr=stds, capsize=2, alpha=0.8)

    plt.xlabel('Protein Family', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.title('Property Distribution by Family', fontsize=16)
    plt.xticks(x_pos + bar_width * 1.5, families, rotation=45, ha='right')
    plt.legend(title='Properties', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/static/Property_Distributions.png', dpi=300)
    plt.close()
    logging.info("Saved figures/static/Property_Distributions.png")


def main():
    logging.info("Starting Static Visualization Pipeline...")
    
    df = load_data()
    if df.empty:
        logging.error("No data to process. Exiting.")
        return

    df = calculate_additional_properties(df)
    df = assign_protein_family(df)
    df = assign_chemical_type(df)
    
    create_ligand_properties_figure(df)
    
    df.to_csv('data/processed/Integrated_Physicochemical_Dataset.csv')
    logging.info("Consolidated dataset exported to data/processed/Integrated_Physicochemical_Dataset.csv")
    print("Static visualization generation complete.")

if __name__ == "__main__":
    main()
