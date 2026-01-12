"""
literature_mining.py

This module performs literature mining using the Europe PMC API to identify chemical compounds
associated with specific terms (e.g., protein targets). It retrieves annotations,
extracts ChEBI IDs, and compiles publication metadata.

Dependencies: urllib, concurrent.futures
"""

import urllib.request
import urllib.parse
import json
import logging
import time
import os
import concurrent.futures
from datetime import datetime


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/literature_mining.log', mode='w')
    ]
)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ANNOTATION_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/med/{}/annotations/1"

def fetch_data(url):
    """
    Performs a GET request to the specified URL and returns the JSON response.
    
    Args:
        url (str): The URL to data from.
        
    Returns:
        dict: The parsed JSON data.
    """
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return None

def get_annotations(pmcid, source):
    """
    Retrieves chemical annotations for a given Publication ID (PMCID/MED).
    
    Args:
        pmcid (str): The publication ID.
        source (str): The source of the ID (e.g., 'MED').
        
    Returns:
        list: A list of chemical names found in the publication.
    """
    try:
        url = ANNOTATION_URL.format(pmcid)
        data = fetch_data(url)
        
        chemicals = []
        if data:
            for annotation in data.get('providers', []):
                for item in annotation.get('annotations', []):
                    if item.get('type') == 'Chemicals':
                        chemicals.append(item.get('exact'))
                        
        return list(set(chemicals))
    except Exception as e:
        logging.warning(f"Error retrieving annotations for {pmcid}: {e}")
        return []

def process_result_batch(results):
    """
    Processes a batch of search results to extract publication metadata and chemicals.
    
    Args:
        results (list): List of publication result objects.
        
    Returns:
        list: A list of dictionaries containing processed data.
    """
    processed_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_pub = {}
        
        for result in results:
            pmid = result.get('id')
            source = result.get('source')
            
            if result.get('hasTextMinedTerms') == 'Y':
                pub_date = result.get('pubYear') or result.get('firstPublicationDate')
                
                future = executor.submit(get_annotations, pmid, source)
                future_to_pub[future] = {
                    'pmid': pmid,
                    'year': pub_date,
                    'source': source
                }
        
        for future in concurrent.futures.as_completed(future_to_pub):
            pub_info = future_to_pub[future]
            try:
                chemicals = future.result()
                if chemicals:
                    processed_data.append({
                        'PMID': pub_info['pmid'],
                        'Year': pub_info['year'],
                        'Chemicals': chemicals
                    })
            except Exception as e:
                logging.error(f"Error processing publication {pub_info['pmid']}: {e}")
                
    return processed_data

def search_publications(query, page_size=25):
    """
    Searches for publications matching the query and orchestrates the extraction process.
    
    Args:
        query (str): The search term.
        page_size (int): results per page.
        
    Returns:
        list: All processed publication data.
    """
    logging.info(f"Searching for: {query}")
    encoded_query = urllib.parse.quote(f'"{query}" AND (HAS_TEXT_MINED_TERMS:y)')
    
    cursor_mark = "*"
    next_cursor_mark = "*"
    
    all_data = []
    page_counter = 0
    max_pages = 20 
    
    while True:
        if page_counter >= max_pages:
            break
            
        url = f"{BASE_URL}?query={encoded_query}&format=json&pageSize={page_size}&cursorMark={cursor_mark}"
        data = fetch_data(url)
        
        if not data:
            break
            
        result_list = data.get('resultList', {}).get('result', [])
        if not result_list:
            break
            
        batch_data = process_result_batch(result_list)
        all_data.extend(batch_data)
        
        next_cursor_mark = data.get('nextCursorMark')
        if cursor_mark == next_cursor_mark:
            break
            
        cursor_mark = next_cursor_mark
        page_counter += 1
        
        if page_counter % 5 == 0:
            logging.info(f"Processed {page_counter} pages...")
            
    return all_data

def save_results(term, data, output_folder):
    """
    Saves the processed data to a JSON file.
    
    Args:
        term (str): The search query term.
        data (list): The data to save.
        output_folder (str): Directory to save in.
    """
    filename = f"{term.replace(' ', '_')}_results.json"
    path = os.path.join(output_folder, filename)
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {len(data)} records to {path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def main():
    logging.info("Literature Mining Module Started")
    
    queries = []
    if os.path.exists('queries.txt'):
        with open('queries.txt', 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        logging.warning("queries.txt not found. Using default query.")
        queries = ["Protein Kinase"]
        
    output_folder = 'data/interim'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for query in queries:
        results = search_publications(query)
        save_results(query, results, output_folder)
        
    logging.info("Literature Mining Completed")

if __name__ == "__main__":
    main()