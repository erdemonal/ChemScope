"""
fetch_resources.py

This module handles the retrieval of essential data dependencies from the OSF repository.
It ensures that all necessary chemical property files and relations are downloaded
and validated before analysis proceeds.

Dependencies: requests, tqdm
"""

import os
import shutil
import json
import collections
import logging
import requests
import tqdm


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fetch_resources.log', mode='w')
    ]
)

EXTENDED_PROPERTY_FILES = [
    'ChEBI2Role_relation.tsv',
    'ChEBI2Application_relation.tsv', 
    'ChEBI2BiologicalRole_relation.tsv',
    'ChEBI2HBD_relation.tsv',
    'ChEBI2HBA_relation.tsv',
    'ChEBI2PSA_relation.tsv',
    'ChEBI2RotBonds_relation.tsv',
    'ChEBI2SMILES_relation.tsv',
    'ChEBI2Pharmacology_relation.tsv',
    'ChEBI2Drug_relation.tsv'
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_files(file_to_link, folder):
    """
    Downloads specified files from OSF to the target directory.

    Args:
        file_to_link (dict): Mapping of filename to download URL.
        folder (str): Target directory.
    """
    logging.info(f"Starting download of {len(file_to_link)} files...")
    
    for file_name, link in file_to_link.items():
        try:
            r = requests.get(link, stream=True, headers=HEADERS)
            r.raise_for_status()
            
            file_size = int(r.headers.get('Content-Length', 0))
            path = os.path.join(folder, file_name)

            desc = f'Downloading {file_name}'
            with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
                with open(path, "wb") as f:
                    shutil.copyfileobj(r_raw, f)
            logging.info(f"Successfully downloaded: {file_name}")
            
        except Exception as e:
            logging.error(f"Error downloading {file_name}: {e}")
    

def get_response(url):
    """
    Fetches JSON data from a URL with error handling.

    Args:
        url (str): The URL to request.

    Returns:
        dict: Parsed JSON response.
    """
    try:
        response = requests.get(url, timeout=30, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        raise


def get_osf_to_rel(url):
    """
    Maps OSF files to their release/version information.

    Args:
        url (str): The OSF API URL.

    Returns:
        dict: Metadata of available files.
    """
    response = get_response(url)
    osf_to_rel = collections.defaultdict(dict)

    for obj in response.get("data", []):
        file_name = obj["attributes"]["name"]
        download_link = obj["links"]["download"]
        file_size = obj["attributes"].get("size", 0)

        if "_" in file_name:
            parts = file_name.split("_")
            file_key = parts[0]
            rel = parts[1].split(".")[0]
        else:
            rel = ""
            file_key = file_name.split(".")[0]
            
        osf_to_rel[file_key] = {
            "rel": rel,
            "link": download_link,
            "name": file_name,
            "size": file_size
        }

    return osf_to_rel


def get_repo_to_rel(folder):
    """
    Maps existing local files to their release/version information.

    Args:
        folder (str): Directory to scan.

    Returns:
        dict: Metadata of local files.
    """
    if not os.path.exists(folder):
        return collections.defaultdict(dict)
        
    repo_to_rel = collections.defaultdict(dict)

    for file_name in os.listdir(folder):
        try:
            if "_" in file_name:
                parts = file_name.split("_")
                file_key = parts[0]
                rel = parts[1].split(".")[0]
            else:
                rel = ""
                file_key = file_name.split(".")[0]
                
            path = os.path.join(folder, file_name)
            
            repo_to_rel[file_key] = {
                "rel": rel,
                "path": path,
                "size": os.path.getsize(path)
            }
        except Exception as e:
            logging.warning(f"Error processing existing file {file_name}: {e}")
            
    return repo_to_rel


def get_files_to_download(osf_to_rel, repo_to_rel):
    """
    Determines which files are missing or outdated.

    Returns:
        dict: Map of filename -> download URL.
    """
    files_to_download = {}
    
    for file_key, osf_data in osf_to_rel.items():
        rel = osf_data['rel']
        link = osf_data['link']
        name = osf_data['name']
        size = osf_data['size']

        should_download = False
        reason = ""

        if file_key in repo_to_rel:
            local_data = repo_to_rel[file_key]
            if rel != local_data['rel']:
                should_download = True
                reason = f"newer release ({rel} vs {local_data['rel']})"
                try:
                    os.remove(local_data['path'])
                    logging.info(f"Removed outdated file: {local_data['path']}")
                except OSError as e:
                    logging.warning(f"Could not remove old file: {e}")
            elif abs(size - local_data['size']) > 1000:
                should_download = True
                reason = f"size mismatch ({size} vs {local_data['size']})"
        else:
            should_download = True
            reason = "new file"

        if should_download:
            files_to_download[name] = link
            logging.info(f"Scheduled for download: {name} ({reason})")

    if not files_to_download:
        logging.info("All files are up to date.")

    return files_to_download


def validate_downloaded_files(folder):
    """
    Validates that core files exist after download.
    """
    if not os.path.exists(folder):
        logging.warning(f"Download folder {folder} does not exist")
        return False
    
    files = os.listdir(folder)
    core_files = ['ChEBI2Mass', 'ChEBI2logP', 'ChEBI2Names']
    missing_core = [f for f in core_files if not any(f in fn for fn in files)]
    
    if missing_core:
        logging.warning(f"Missing core files: {missing_core}")
        return False
    
    found = [f for f in EXTENDED_PROPERTY_FILES if any(f.replace('_relation.tsv','') in fn for fn in files)]
    logging.info(f"Core files valid. Extended properties found: {len(found)}")
    
    return True


def main():
    logging.info("ChemScope File Downloader initialized.")
    
    folder = 'data/raw'
    url = 'https://api.osf.io/v2/nodes/pvwu2/files/osfstorage/611252ba847d1304ca38b4d4/'

    if not os.path.isdir(folder):
        os.makedirs(folder)
        logging.info(f"Created directory: {folder}")

    logging.info("Querying OSF repository...")
    osf_to_rel = get_osf_to_rel(url)
    repo_to_rel = get_repo_to_rel(folder)

    logging.info(f"Repository files: {len(osf_to_rel)} | Local files: {len(repo_to_rel)}")

    files_to_download = get_files_to_download(osf_to_rel, repo_to_rel)

    if files_to_download:
        download_files(files_to_download, folder)
    
    logging.info("Validating downloads...")
    if validate_downloaded_files(folder):
        print("\nDownload completed successfully.")
    else:
        print("\nDownload completed with warnings. Some files may be missing.")


if __name__ == '__main__':
    main()