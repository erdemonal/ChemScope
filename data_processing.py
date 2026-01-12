"""
data_processing.py

This module handles the preprocessing of chemical data extracted from literature.
It integrates chemical properties from ChEBI, calculates additional physicochemical
descriptors using RDKit, and assigns functional roles (e.g., Ligand, Buffer, Solvent)
based on algorithmic criteria.

The output consists of processed datasets suitable for subsequent chemometric analysis.

Dependencies: pandas, numpy, rdkit
"""

import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_processing.log', mode='w')
    ]
)

DEFAULT_STOP_LIST = [
    '15377', 
    '26710', 
    '9754',  
    '42334', 
    '28262', 
    '16236', 
    '17883', 
    '15347', 
    '27385', 
    '9344',  
    '16810', 
    '26836', 
]


def determine_chemical_role(row: pd.Series) -> str:
    """
    Assigns a functional role (Ligand, Buffer, Solvent, etc.) to a chemical entity
    based on its physicochemical properties and nomenclature.

    Args:
        row (pd.Series): A row containing chemical properties (Mass, logP, Names).

    Returns:
        str: The assigned role.
    """
    try:
        mass = float(row.get('Mass', 0)) if pd.notna(row.get('Mass')) else 0
        logp = float(row.get('logP', 0)) if pd.notna(row.get('logP')) else 0
    except (ValueError, TypeError):
        mass, logp = 0, 0
        
    name = str(row.get('Names', '')).lower()
    
    buffer_keywords = ['buffer', 'tris', 'hepes', 'phosphate', 'acetate', 'citrate', 'chloride', 'sulfate']
    solvent_keywords = ['water', 'dmso', 'ethanol', 'methanol', 'acetonitrile', 'chloroform']
    
    if any(k in name for k in buffer_keywords):
        return 'Buffer'
    if any(k in name for k in solvent_keywords):
        return 'Solvent'

    if mass < 150:
        if logp < -1:
            return 'Buffer/Salt'
        else:
            return 'Solvent/Reagent'
            
    if 150 <= mass <= 900:
        if -2 <= logp <= 6:
            return 'Ligand'
            
    return 'Potential Ligand'


def calculate_rdkit_properties(smiles: str):
    """
    Calculates extended molecular properties using RDKit software.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        tuple: (HBD, HBA, TPSA, RotBonds, Mass, logP)
    """
    if pd.isna(smiles) or str(smiles).strip() in ['-', '', 'nan']:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            logp = Descriptors.MolLogP(mol)
            mass = Descriptors.ExactMolWt(mol)
            return hbd, hba, tpsa, rot_bonds, mass, logp
    except Exception:
        pass
    
    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def load_stop_list(stop_list_path=None):
    """
    Loads the stop list from a file or returns the default list.
    """
    if stop_list_path and os.path.exists(stop_list_path):
        with open(stop_list_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return DEFAULT_STOP_LIST


def apply_stop_list_filter(df_results, stop_list):
    """
    Filters out ChEBI IDs present in the stop list.
    """
    stop_ids = []
    for id_str in stop_list:
        try:
            stop_ids.append(int(id_str))
        except ValueError:
            logging.warning(f"Invalid ChEBI ID in stop list: {id_str}")
    
    initial_count = len(df_results)
    df_results = df_results[~df_results.index.isin(stop_ids)]
    removed_count = initial_count - len(df_results)
    
    if removed_count > 0:
        logging.info(f"Stop list filter removed {removed_count} common non-ligands")
    
    return df_results


def import_properties():
    '''
    Reads ChEBI property files from the 'data/raw' directory.
    '''
    FOLDER = 'data/raw'
    if not os.path.exists(FOLDER):
        logging.error(f"{FOLDER} directory not found. Please run fetch_resources.py first.")
        return {}
    
    files = os.listdir(FOLDER)
    data = dict()
    
    property_mapping = {
        'role': 'Role',
        'application': 'Application',
        'hbd': 'HBD',
        'hba': 'HBA',
        'psa': 'PSA',
        'rotbonds': 'RotBonds',
        'rotatable': 'RotBonds',
        'smiles': 'SMILES'
    }
    
    for file in files:
        path = os.path.join(FOLDER, file)
        
        key = None
        file_lower = file.lower()
        
        for prop_identifier, prop_key in property_mapping.items():
            if prop_identifier in file_lower:
                key = prop_key
                break
        
        if key is None:
            if '_' in file and '2' in file:
                try:
                    key = file.split('2')[1].split('_')[0]
                except IndexError:
                    key = file.split('.')[0]
            else:
                key = file.split('.')[0]
        
        try:
            if '.pkl' in file:
                df = pd.read_pickle(path)
                if 'ChEBI' in df.columns:
                    df['ChEBI'] = df['ChEBI'].astype(int)
                    df = df.set_index('ChEBI')
            else:
                try:
                    df = pd.read_csv(path, sep='\t', header=None, 
                                   names=['ChEBI', 'Info'], index_col='ChEBI',
                                   dtype={"ChEBI": "int", "Info": "str"})
                except ValueError:
                    df = pd.read_csv(path, sep='\t', index_col=0)
                    if len(df.columns) == 1:
                        df.columns = ['Info']
            
            if not df.empty:
                data[key] = df.iloc[:, 0].to_dict()
                
            logging.info(f"Loaded {len(df)} entries for property: {key}")
            
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
            continue
    
    return data


def import_publication_dates(term):
    '''
    Imports publication dates for a specific search term.
    Returns:
        dict: Mapping of PublicationID to PublicationYear.
    '''
    pub_dates_dict = {}
    pub_dates_file = os.path.join('data/metadata/publication_dates', f'{term}_pub_dates.tsv')
    
    if os.path.exists(pub_dates_file):
        try:
            pub_dates_df = pd.read_csv(pub_dates_file, sep='\t')
            pub_dates_dict = dict(zip(pub_dates_df['PublicationID'], pub_dates_df['PublicationYear']))
        except Exception:
            pass
    
    return pub_dates_dict


def process_table(data, df_results, publication_dates=None, use_stop_list=True, stop_list_path=None):
    '''
    Constructs the data table by integrating properties and calculating descriptors.
    '''
    table = df_results.copy()
    
    if use_stop_list:
        stop_list = load_stop_list(stop_list_path)
        table = apply_stop_list_filter(table, stop_list)

    for key in data.keys():
        table[key] = table.index.map(data[key])
        
        if key in ['Mass', 'logP']:
            table[key] = pd.to_numeric(table[key].replace('-', np.nan), errors='coerce')


    logging.info("Computing chemical descriptors via RDKit...")
    
    hbds, hbas, tpsas, rots = [], [], [], []
    
    if 'SMILES' in table.columns:
        for idx in table.index:
            smi = table.loc[idx, 'SMILES']
            hbd, hba, tpsa, rot, r_mass, r_logp = calculate_rdkit_properties(smi)
            
            hbds.append(hbd)
            hbas.append(hba)
            tpsas.append(tpsa)
            rots.append(rot)
            
            if pd.isna(table.loc[idx, 'Mass']) and pd.notna(r_mass):
                table.loc[idx, 'Mass'] = r_mass
            if pd.isna(table.loc[idx, 'logP']) and pd.notna(r_logp):
                table.loc[idx, 'logP'] = r_logp
            
        table['HBD'] = hbds
        table['HBA'] = hbas
        table['TPSA'] = tpsas
        table['RotBonds'] = rots
    else:
        logging.warning("SMILES column not found. RDKit property calculation skipped.")

    logging.info("Assigning chemical roles algorithmically...")
    table['Role'] = table.apply(determine_chemical_role, axis=1)

    if 'idf' in table.columns:
        table["Count"] = pd.to_numeric(table["Count"], errors='coerce')
        table["idf"] = pd.to_numeric(table["idf"], errors='coerce')
        tfidf_values = table["Count"].astype(float) * table["idf"].astype(float)
        tfidf_values = tfidf_values.fillna(0).replace([np.inf, -np.inf], 0)
        table.loc[:,"TFIDF"] = tfidf_values.round(decimals=3)

    return table


def write_to_file(table, term):
    '''
    Writes the processed data table to disk in TSV and Pickle formats.
    '''
    os.makedirs('data/processed', exist_ok=True)
    path = f'data/processed/{term}'
    
    try:
        cols = ['Names', 'Count', 'Role', 'Mass', 'logP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'SMILES', 'TFIDF']
        cols = [c for c in cols if c in table.columns]
        remaining = [c for c in table.columns if c not in cols]
        
        table[cols + remaining].to_csv(f'{path}_Binding_Dataset.tsv', sep='\t')
        table.to_pickle(f'{path}_Dataset.pkl')
        
        summary = {
            'Total Compounds': len(table),
            'Ligands': len(table[table['Role'] == 'Ligand']),
            'Buffers': len(table[table['Role'] == 'Buffer']),
            'Solvents': len(table[table['Role'] == 'Solvent']),
        }
        
        with open(f'{path}_Dataset_Summary.txt', 'w') as f:
            for k,v in summary.items():
                f.write(f"{k}: {v}\n")
                
        logging.info(f"Files written successfully: {path}_Binding_Dataset.tsv")
        
    except Exception as e:
        logging.error(f"Error writing files for {term}: {str(e)}")
        table.to_pickle(f'{path}_Dataset.pkl')


def parser():
    parser = argparse.ArgumentParser(description='Enhanced table creation with filtering options')
    parser.add_argument('-i', required=True, metavar='input', dest='input', 
                       help='Input folder or file from the data/interim folder')
    parser.add_argument('-t', required=True, metavar='type', dest='type', 
                       help='Type of input: file or folder')
    parser.add_argument('--no-stop-list', action='store_true',
                       help='Disable default stop list filtering')
    parser.add_argument('--stop-list', type=str,
                       help='Path to custom stop list file')
    
    return parser.parse_args()


def main():
    args = parser()
    input_type = args.type
    input_path = args.input
    
    if input_type == 'file':
        results = [input_path]
    elif input_type == 'folder':
        if not os.path.exists(input_path):
            logging.critical(f'Input folder {input_path} does not exist')
            sys.exit(1)
        files = os.listdir(input_path)
        results = [f'{input_path}/{file}' for file in files if file.endswith('.tsv')]
    else:
        logging.critical('Invalid input type. Use "file" or "folder".')
        sys.exit(1)

    if not results:
        logging.critical(f'No TSV files found in {input_path}')
        sys.exit(1)

    logging.info("Loading ChEBI properties...")
    data = import_properties()
    
    logging.info(f"Loaded properties: {list(data.keys())}")

    for result in results:
        try:
            filename = os.path.basename(result)
            if '_ChEBI_IDs.tsv' in filename:
                term = filename.replace('_ChEBI_IDs.tsv', '')
            else:
                term = filename.replace('.tsv', '')
                
            logging.info(f'Processing dataset: {term}...')

            publication_dates = import_publication_dates(term)

            df = pd.read_csv(result, sep='\t', names=['ChEBI', 'Publication'], 
                           dtype={"ChEBI": "int", "Publication": "str"})
            df_results = df.groupby(by=['ChEBI']).count().rename(columns={"Publication": "Count"})

            logging.info(f"Initial compounds found: {len(df_results)}")

            table = process_table(data, df_results, publication_dates,
                                use_stop_list=not args.no_stop_list,
                                stop_list_path=args.stop_list)

            logging.info(f"Final compounds after processing: {len(table)}")

            write_to_file(table, term)
            
        except Exception as e:
            logging.error(f"Error processing {result}: {str(e)}")
            continue

    logging.info("Data processing completed successfully.")

if __name__ == '__main__':
    main()