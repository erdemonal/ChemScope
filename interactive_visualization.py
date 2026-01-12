"""
interactive_visualization.py

This module generates interactive localization plots (using Bokeh) to visualize the
chemical space of protein-ligand interactions. It maps chemicals based on their
Molecular Weight and LogP, utilizing hexagonal binning statistics.

It supports advanced features such as:
- Gaussian blurring for density estimation.
- Interactive filtering by chemical class and properties.
- TF-IDF weighting for relevance scoring.
- Drug-likeness analysis (Lipinski, Veber rules).

Dependencies: bokeh, pandas, numpy
"""

import time
import argparse
import logging
import os
from math import exp, sqrt
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from bokeh import events
from bokeh.io import output_file, show
from bokeh.models import (
    CustomJS, HoverTool, ColumnDataSource, Slider, CheckboxGroup, 
    RadioGroup, Button, MultiSelect
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import cartesian_to_axial


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/visualization.log', mode='w')
    ]
)

def import_table(file_path):
    """
    Imports a pickle file containing chemical data.

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        table = pd.read_pickle(file_path)
        return table
    except Exception as e:
        logging.error(f"Failed to import table {file_path}: {e}")
        return pd.DataFrame()


def create_array(table):
    """
    Extracts Mass and logP arrays from the DataFrame for plotting.

    Args:
        table (pd.DataFrame): Input data containing 'Mass' and 'logP' columns.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray) -> (logP, Mass)
    """
    valid_data = table.dropna(subset=['Mass', 'logP'])
    
    if len(valid_data) == 0:
        logging.warning("No compounds with complete Mass and logP data found.")
        return np.array([]), np.array([])
    
    x = [float(logP) for logP in valid_data.logP]
    y = [float(mass) for mass in valid_data.Mass]
    
    logging.debug(f"Created arrays for {len(x)} compounds.")
    
    return np.asarray(x), np.asarray(y)


def hexbin(df, x, y, size, aspect_scale, orientation):
    """
    Converts Cartesian coordinates (x, y) to Axial hexagonal coordinates (q, r).

    Args:
        df (pd.DataFrame): Dataframe to append coordinates to.
        x (array-like): X (logP) coordinates.
        y (array-like): Y (Mass) coordinates.
        size (float): Hexagon size.
        aspect_scale (float): Aspect ratio scaling.
        orientation (str): Hexagon orientation.

    Returns:
        pd.DataFrame: Dataframe including 'q' and 'r' columns.
    """
    q, r = cartesian_to_axial(x, y, size, orientation=orientation, aspect_scale=aspect_scale)
    df.loc[:, 'q'] = q
    df.loc[:, 'r'] = r
    return df


def add_tooltip_columns(df, table):
    """
    Enriches the dataset with tooltip information (e.g., top names per hexagon).

    Args:
        df (pd.DataFrame): Grouped hexagon data.
        table (pd.DataFrame): Original detailed data.

    Returns:
        pd.DataFrame: Dataframe with added tooltip columns.
    """
    table = table.drop(['Class', 'logP', 'Mass'], axis=1, errors='ignore')
    table = table.reset_index()

    TOOLTIP_COUNT = 3
    columns = table.columns
    tooltip_columns = {f"{col}{i}": [] for i in range(1, TOOLTIP_COUNT + 1) for col in columns}

    chebi_ids_list = [ids if isinstance(ids, list) else [] for ids in df.ChEBI]
    list_for_df = []

    for ids in chebi_ids_list:
        if not ids:
            list_for_df.append(["-"] * (TOOLTIP_COUNT * len(columns)))
            continue

        rows = table[table.ChEBI.isin(ids)]
        rows = rows.sort_values(by='Count', ascending=False)
        
        values_nested = rows[0:TOOLTIP_COUNT].values.tolist()
        values_unnested = [item for sublist in values_nested for item in sublist]
        
        while len(values_unnested) < (TOOLTIP_COUNT * len(columns)):
            values_unnested.append("-")

        list_for_df.append(values_unnested)

    df_tooltip = pd.DataFrame(list_for_df, columns=tooltip_columns.keys())
    df = df.join(df_tooltip, how='left')

    return df


def get_blur(x, y, sigma_x, sigma_y):
    """
    Calculates the Gaussian blur intensity.
    """
    return exp(-0.5 * (x * x / (sigma_x * sigma_x) + y * y / (sigma_y * sigma_y)))


def get_rows(q, r, counts, tfidf, kernel, blur_max, step_size):
    """
    Distributes counts/scores to neighboring hexagons based on a Gaussian kernel.
    """
    base_row = [q, r, counts, tfidf] + \
               [counts for _ in np.arange(0, blur_max + step_size, step_size)] + \
               [tfidf for _ in np.arange(0, blur_max + step_size, step_size)]
    
    rows = [base_row]

    for coords_new, blur_factors in kernel.items():
        q_new = q + coords_new[0]
        r_new = r + coords_new[1]

        new_row = [q_new, r_new, 0, 0] + \
                  list(map(lambda n: n * counts, blur_factors)) + \
                  list(map(lambda n: n * tfidf, blur_factors))
        rows.append(new_row)
        
    return rows


def construct_kernel(blur_max, step_size):
    """
    Constructs the Gaussian kernel for hexagonal grids.

    Returns:
        dict: Mapping of relative coordinates to blur factors.
    """
    coordinates_to_distance = {
        (-5, 2): (7.5, sqrt(3)/2), (-5, 3): (7.5, sqrt(3)/2),
        (-4, 1): (6, sqrt(3)), (-4, 2): (6, 0), (-4, 3): (6, sqrt(3)),
        (-3, 0): (4.5, 3*sqrt(3)/2), (-3, 1): (4.5, sqrt(3)/2), (-3, 2): (4.5, sqrt(3)/2), (-3, 3): (4.5, 3*sqrt(3)/2),
        (-2, -1): (3, 2*sqrt(3)), (-2, 0): (3, sqrt(3)), (-2, 1): (3, 0), (-2, 2): (3, sqrt(3)), (-2, 3): (3, 2*sqrt(3)),
        (-1, -1): (1.5, 3*sqrt(3)/2), (-1, 0): (1.5, sqrt(3)/2), (-1, 1): (1.5, sqrt(3)/2), (-1, 2): (1.5, 3*sqrt(3)/2),
        (0, -2): (0, 2*sqrt(3)), (0, -1): (0, sqrt(3)), (0, 1): (0, sqrt(3)), (0, 2): (0, 2*sqrt(3)),
        (1, -2): (1.5, 3*sqrt(3)/2), (1, -1): (1.5, sqrt(3)/2), (1, 0): (1.5, sqrt(3)/2), (1, 1): (1.5, 3*sqrt(3)/2),
        (2, -3): (3, 2*sqrt(3)), (2, -2): (3, sqrt(3)), (2, -1): (3, 0), (2, 0): (3, sqrt(3)), (2, 1): (3, 2*sqrt(3)),
        (3, -3): (4.5, 3*sqrt(3)/2), (3, -2): (4.5, sqrt(3)/2), (3, -1): (4.5, sqrt(3)/2), (3, 0): (4.5, 3*sqrt(3)/2),
        (4, -3): (6, sqrt(3)), (4, -2): (6, 0), (4, -1): (6, sqrt(3)),
        (5, -3): (7.5, sqrt(3)/2), (5, -2): (7.5, sqrt(3)/2)
    }

    kernel = {}
    for key, distance in coordinates_to_distance.items():
        kernel[key] = [0]
        for sd_x in np.arange(step_size, blur_max + step_size, step_size):
            kernel[key].append(get_blur(distance[0], distance[1], sd_x, sd_x/2))

    return kernel


def add_gaussian_blur(df, blur_max, step_size):
    """
    Applies Gaussian blur to the hexagonal grid data.
    """
    kernel = construct_kernel(blur_max, step_size)

    sd_range = np.arange(0, (blur_max + step_size), step_size)
    columns = ['q', 'r', 'Count', 'TFIDF'] + \
              [str(sd_x) for sd_x in sd_range] + \
              [f'{sd_x}_tfidf' for sd_x in sd_range]

    expanded_rows = []
    for q, r, counts, tfidf in zip(df.q, df.r, df.Count, df.TFIDF):
        expanded_rows.extend(get_rows(q, r, counts, tfidf, kernel, blur_max, step_size))
        
    df_blur = pd.DataFrame(expanded_rows, columns=columns)

    df_blur = df_blur.groupby(['q', 'r'], as_index=False).sum()

    df_joined = df_blur.merge(df.loc[:, ['q', 'r', 'ChEBI']], on=['q', 'r'], how='outer')

    return df_joined


def check_for_ids(id_list, class_id):
    """
    Checks if a specific Class ID exists in a list of IDs.
    """
    return any(str(class_id) == str(id) for id in id_list)


def analyze_publication_years(table):
    """
    Computes statistical metrics for publication years.
    """
    years_data = []
    
    if 'PublicationYears' in table.columns:
        for years_list in table['PublicationYears']:
            if isinstance(years_list, list) and years_list:
                years_data.extend(years_list)
    
    if not years_data:
        return {'has_data': False}
    
    return {
        'has_data': True,
        'min_year': min(years_data),
        'max_year': max(years_data),
        'mean_year': round(np.mean(years_data), 2),
        'median_year': np.median(years_data),
        'year_counts': {str(k): v for k, v in sorted(Counter(years_data).items())}
    }


def analyze_drug_likeness(table):
    """
    Computes drug-likeness compliance statistics.
    """
    drug_stats = {}
    
    def calc_stat(col_name):
        if col_name in table.columns:
            count = table[col_name].sum()
            total = table[col_name].count()
            pct = round((count / total) * 100, 1) if total > 0 else 0
            return count, pct
        return None, None

    lip_c, lip_p = calc_stat('Lipinski_RO5')
    if lip_c is not None:
        drug_stats['lipinski_compliant'] = lip_c
        drug_stats['lipinski_percentage'] = lip_p

    veb_c, veb_p = calc_stat('Veber_Rule')
    if veb_c is not None:
        drug_stats['veber_compliant'] = veb_c
        drug_stats['veber_percentage'] = veb_p

    dl_c, dl_p = calc_stat('Drug_Like')
    if dl_c is not None:
        drug_stats['drug_like'] = dl_c
        drug_stats['drug_like_percentage'] = dl_p
    
    return drug_stats


def create_class_source(table, size, ratio, orientation, class_id):
    """
    Creates a ColumnDataSource for highlighting a specific chemical class.
    """
    if class_id is None:
        return ColumnDataSource(pd.DataFrame(columns=['q', 'r']))

    mask = [check_for_ids(ids, class_id) for ids in table["Class"]]
    table_class_only = table[mask]

    if table_class_only.empty:
         return ColumnDataSource(pd.DataFrame(columns=['q', 'r']))

    x, y = create_array(table_class_only)
    q, r = cartesian_to_axial(x, y, size, orientation=orientation, aspect_scale=ratio)
    
    df = table_class_only.reset_index()
    df.loc[:, "q"] = q
    df.loc[:, "r"] = r

    df = df.groupby(['q', 'r']).agg({'Count': 'sum', 'TFIDF': 'sum'}).reset_index()
    
    return ColumnDataSource(df)


def create_drug_like_source(table, size, ratio, orientation, rule_type='Drug_Like'):
    """
    Creates a ColumnDataSource for highlighted drug-like compounds.
    """
    if rule_type not in table.columns:
        return ColumnDataSource(pd.DataFrame(columns=['q', 'r', 'Count']))
    
    drug_like_table = table[table[rule_type] == True]
    
    if drug_like_table.empty:
        return ColumnDataSource(pd.DataFrame(columns=['q', 'r', 'Count']))
    
    x, y = create_array(drug_like_table)
    if len(x) == 0:
        return ColumnDataSource(pd.DataFrame(columns=['q', 'r', 'Count']))
        
    q, r = cartesian_to_axial(x, y, size, orientation=orientation, aspect_scale=ratio)
    df = drug_like_table.reset_index()
    df.loc[:, "q"] = q
    df.loc[:, "r"] = r

    df = df.groupby(['q', 'r']).agg({'Count': 'sum'}).reset_index()
    
    return ColumnDataSource(df)


def create_data_source(table, term, size, ratio, orientation, BLUR_MAX, BLUR_STEP_SIZE):
    """
    Constructs the main ColumnDataSource for the hexagonal plot.
    """
    table_filtered = table.dropna(subset=['Mass', 'logP']).copy()
    
    if len(table_filtered) == 0:
        logging.warning(f"No compounds with complete Mass/logP data for {term}")
        empty_df = pd.DataFrame(columns=['q', 'r', 'Count', 'TFIDF'])
        return ColumnDataSource(empty_df), f'No data available for {term}'
    
    logging.info(f"Using {len(table_filtered)} compounds for {term}")
    
    x = [float(logP) for logP in table_filtered.logP]
    y = [float(mass) for mass in table_filtered.Mass]
    
    q, r = cartesian_to_axial(np.asarray(x), np.asarray(y), size, orientation=orientation, aspect_scale=ratio)

    df = table_filtered.reset_index(drop=False)
    df = df.iloc[:len(q)]
    df.loc[:, "q"] = q
    df.loc[:, "r"] = r

    agg_dict = {'Count': 'sum', 'ChEBI': list}
    if 'TFIDF' in df.columns:
        agg_dict['TFIDF'] = 'sum'
    else:
        df['TFIDF'] = df['Count']
        agg_dict['TFIDF'] = 'sum'
    
    df_grouped = df.groupby(['q', 'r']).agg(agg_dict).reset_index()
    
    try:
        df_blurred = add_gaussian_blur(df_grouped, BLUR_MAX, BLUR_STEP_SIZE)
        df_final = add_tooltip_columns(df_blurred, table_filtered)
    except Exception as e:
        logging.warning(f"Blurring failed ({e}), using raw data.")
        df_final = df_grouped.copy()
        for sd_x in np.arange(0, BLUR_MAX + 0.25, 0.25):
            df_final[str(sd_x)] = df_final['Count']
            if f'{sd_x}_tfidf' not in df_final.columns:
                df_final[f'{sd_x}_tfidf'] = df_final['TFIDF']
    
    if 'ChEBI' in df_final.columns:
        df_final = df_final.drop(columns='ChEBI', errors='ignore')
    
    df_final.loc[:, "Count_total"] = df_final.loc[:, "Count"]
    
    for col in ['Count', 'TFIDF', 'Count_total']:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

    title = f'Hexbin plot for {len(x)} compounds with query {term}'
    return ColumnDataSource(df_final), title


def create_stats_description(table):
    """
    Generates an HTML description of statistical properties for the dataset.
    """
    total_count = table.Count.sum()
    
    stats_listed = [
        'Chemical Property Statistics',
        f'Total amount of chemicals: {total_count}',
    ]
    
    properties = {
        'logP': 'LogP',
        'Mass': 'Molecular Weight (Da)',
        'HBD': 'Hydrogen Bond Donors',
        'HBA': 'Hydrogen Bond Acceptors',
        'PSA': 'Polar Surface Area',
        'RotBonds': 'Rotatable Bonds'
    }
    
    for prop_col, prop_name in properties.items():
        if prop_col in table.columns:
            prop_data = pd.to_numeric(table[prop_col], errors='coerce').dropna()
            
            if len(prop_data) > 0:
                weighted_data = np.repeat(prop_data, table.loc[prop_data.index, 'Count'])
                stats_listed.extend([
                    f'{prop_name} mean: {weighted_data.mean():.3f}',
                    f'{prop_name} std dev: {weighted_data.std():.3f}',
                    f'{prop_name} median: {np.median(weighted_data):.3f}'
                ])
    
    drug_stats = analyze_drug_likeness(table)
    if drug_stats:
        stats_listed.append('<br><b>Drug-likeness Analysis:</b>')
        for k, v in drug_stats.items():
            if 'percentage' in k: continue
            pct = drug_stats.get(k.replace('_compliant', '_percentage').replace('drug_like', 'drug_like_percentage'), 0)
            label = k.replace('_', ' ').capitalize()
            stats_listed.append(f"{label}: {v} ({pct}%)")

    pub_years = analyze_publication_years(table)
    if pub_years.get('has_data'):
        stats_listed.append('<br><b>Publication Timeline:</b>')
        stats_listed.append(f"Range: {pub_years['min_year']} - {pub_years['max_year']}")
        stats_listed.append(f"Mean: {pub_years['mean_year']}")
        
        stats_listed.append('<br><b>Publications per year:</b>')
        year_dist_html = '<table border="1" cellpadding="3" style="border-collapse: collapse;">'
        year_dist_html += '<tr><th>Year</th><th>Publications</th></tr>'
        for year, count in pub_years['year_counts'].items():
            year_dist_html += f'<tr><td>{year}</td><td>{count}</td></tr>'
        year_dist_html += '</table>'
        stats_listed.append(year_dist_html)

    return return_html(stats_listed)


def return_html(metadata):
    """
    Formats the metadata list into an HTML string.
    """
    html_content = f"""
    <HTML>
    <HEAD>
    <TITLE>{metadata[0]}</TITLE>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.4; }}
        h1 {{ color: #444; }}
        .stats-container {{ margin-top: 20px; }}
        table {{ margin-top: 10px; }}
        th {{ background-color: #e6e6e6; }}
    </style>
    </HEAD>
    <BODY BGCOLOR="FFFFFF">
    <h1>{metadata[0]}</h1>
    <div class="stats-container">
    """
    for item in metadata[1:]:
        html_content += f"<div>{item}</div>\n"
    
    html_content += "</div></BODY></HTML>"
    return html_content


def get_tables(files):
    """
    Loads multiple table files and their associated metadata.
    """
    tables = dict()
    for file in files:
        table = import_table(file)
        if table.empty: continue
        
        term = file.split('/')[-1].split('_Dataset')[0]
        metadata_file = f'data/metadata/{term}.txt'
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = f.readlines()
        except IOError:
            metadata = [f"Metadata for {term}", "No metadata available"]
        
        tables[term] = {'table': table, 'metadata': metadata}

    return tables


def get_files(folder):
    """Returns a list of .pkl files in the specific folder."""
    return [f'{folder}/{file}' for file in os.listdir(folder) if '.pkl' in file and '_Dataset.pkl' in file]


def return_JS_code(widget):
    """
    Returns the JavaScript callback code for a given widget type.
    """
    
    if widget == 'multi_select_class':
        return """
            var source_data = source.data;
            var source_class_data = source_class.data;
            var term_to_source = term_to_source;
            var term = cb_obj.value[0];
            var f = slider1.value
            var sd_x = slider2.value;
            var p = p;
            var mapper = mapper;
            var class_hex = class_hex;
            var checkbox_class = checkbox_class
            var term_to_class = term_to_class;
            
            if (sd_x % 1 == 0){var sd_x = sd_x.toFixed(1)}
            var sd_x = String(sd_x);

            if (checkbox.active.length == 1) {
                sd_x = sd_x.concat('_tfidf')
            }

            var new_data = term_to_source[term]['source'].data
            var new_class = term_to_class[term]['source'].data

            for (var key in new_data) {
                source_data[key] = [];
                for (i=0;i<new_data[key].length;i++) {
                    source_data[key].push(new_data[key][i]);
                }
            }

            class_hex.visible = false
            for (var key in new_class){
                source_class_data[key] = [];
                for (i=0;i<new_class[key].length;i++) {
                    source_class_data[key].push(new_class[key][i])
                }
            }
            class_hex.source = source_class_data
            checkbox_class.active = []

            var title = term_to_source[term]['title']
            p.title.text = title

            for (var i = 0; i < source_data[sd_x].length; i++) {
                source_data['Count'][i] = Math.pow(source_data[sd_x][i], 1/f)
            }

            mapper.transform.high = Math.max.apply(Math, source_data['Count'])
            source_class.change.emit();
            source.change.emit();
        """
        
    elif widget == 'drug_likeness':
        return """
            var source_drug_data = source_drug.data;
            var term_to_drug_sources = term_to_drug_sources;
            var term = multi_select.value[0];
            var active = cb_obj.active;
            var drug_hex = drug_hex;

            if (active.length > 0) {
                var drug_data = term_to_drug_sources[term]['source'].data;
                for (var key in drug_data) {
                    source_drug_data[key] = [];
                    for (i=0; i<drug_data[key].length; i++) {
                        source_drug_data[key].push(drug_data[key][i]);
                    }
                }
                drug_hex.visible = true;
            } else {
                drug_hex.visible = false;
            }
            source_drug.change.emit();
        """
    
    elif widget == 'multi_select':
        return """
            var source_data = source.data;
            var term_to_source = term_to_source;
            var term = cb_obj.value[0];
            var f = slider1.value
            var sd_x = slider2.value;
            var p = p;
            var mapper = mapper;

            if (sd_x % 1 == 0){var sd_x = sd_x.toFixed(1)}
            var sd_x = String(sd_x);

            if (checkbox.active.length == 1) {
                sd_x = sd_x.concat('_tfidf')
            }

            var new_data = term_to_source[term]['source'].data

            for (var key in new_data) {
                source_data[key] = [];
                for (i=0;i<new_data[key].length;i++) {
                    source_data[key].push(new_data[key][i]);
                }
            }

            var title = term_to_source[term]['title']
            p.title.text = title

            for (var i = 0; i < source_data[sd_x].length; i++) {
                source_data['Count'][i] = Math.pow(source_data[sd_x][i], 1/f)
            }

            mapper.transform.high = Math.max.apply(Math, source_data['Count'])
            source.change.emit();
        """
        
    elif widget == 'tooltips':
        return """
            <style>
            table, td, th {
              border-collapse: collapse;
              border: 1px solid #dddddd;
              padding: 2px;
              table-layout: fixed;
              height: 20px;
            }
            tr:nth-child(even) { background-color: #dddddd; }
            </style>
            <table>
              <col width="100"><col width="80"><col width="80"><col width="305">
              <tr><th>Total Counts</th><th>@Count_total</th><th>@TFIDF</th><th>($x, $y)</th></tr>
              <tr style="color: #fff; background: black;"><th>ChEBI ID</th><th>Count</th><th>TFIDF</th><th>Name</th></tr>
              <tr><th>@ChEBI1</th><th>@Count1</th><th>@TFIDF1</th><th>@Names1</th></tr>
              <tr><th>@ChEBI2</th><th>@Count2</th><th>@TFIDF2</th><th>@Names2</th></tr>
              <tr><th>@ChEBI3</th><th>@Count3</th><th>@TFIDF3</th><th>@Names3</th></tr>
            </table>
            """

    elif widget == 'slider2':
        return """
            var source_data = source.data;
            var f = slider1.value
            var mapper = mapper;
            var checkbox = checkbox;
            var sd_x = cb_obj.value;

            if (sd_x % 1 == 0){var sd_x = sd_x.toFixed(1)}
            var sd_x = String(sd_x);

            if (checkbox.active.length == 1) {
                sd_x = sd_x.concat('_tfidf')
            }

            for (var i = 0; i < source_data[sd_x].length; i++) {
                source_data['Count'][i] = Math.pow(source_data[sd_x][i], 1/f)
            }

            mapper.transform.high = Math.max.apply(Math, source_data['Count'])
            source.change.emit();
        """

    elif widget == 'slider1':
        return """
            var mapper = mapper
            var source_data = source.data;
            var f = cb_obj.value;
            var checkbox = checkbox;
            var sd_x = slider2.value;
            if (sd_x % 1 == 0){var sd_x = sd_x.toFixed(1)}
            var sd_x = String(sd_x);

            if (checkbox.active.length == 1) {
                sd_x = sd_x.concat('_tfidf')
            }

            for (var i = 0; i < source_data[sd_x].length; i++) {
                source_data['Count'][i] = Math.pow(source_data[sd_x][i], 1/f)
            }
            mapper.transform.high = Math.max.apply(Math, source_data['Count'])
            source.change.emit();
        """

    elif widget == 'rbg':
        return """
            var active = cb_obj.active;
            var p = p;
            var Viridis256 = Viridis256;
            var Greys256 = Greys256;

            if (active == 1){
                mapper.transform.palette = Greys256
                p.background_fill_color = '#000000'
            }
            if (active == 0){
                mapper.transform.palette = Viridis256
                p.background_fill_color = '#440154'
            }
        """

    elif widget == 'checkbox':
        return """
            var source_data = source.data;
            var active = cb_obj.active
            var f = slider1.value
            var mapper = mapper
            var sd_x = slider2.value;

            if (sd_x % 1 == 0){var sd_x = sd_x.toFixed(1)}
            var sd_x = String(sd_x);

            if (active.length == 1) {
                sd_x = sd_x.concat('_tfidf')
            }

            for (var i = 0; i < source_data[sd_x].length; i++) {
                source_data['Count'][i] = Math.pow(source_data[sd_x][i], 1/f)
            }

            mapper.transform.high = Math.max.apply(Math, source_data['Count'])
            source.change.emit();
        """

    elif widget == 'hover':
        return """
            var tooltips = document.getElementsByClassName("bk-tooltip");
            for (var i = 0, len = tooltips.length; i < len; i ++) {
                tooltips[i].style.top = ""; 
                tooltips[i].style.left = "";
                tooltips[i].style.bottom = "150px";
                tooltips[i].style.left = "575px";
                tooltips[i].style.width = "500px";
            }
        """

    elif widget == 'button':
        return """
            var term_to_metadata = term_to_metadata
            var term = multi_select.value[0]
            var metadata = term_to_metadata[term]
            var wnd = window.open("about:blank", "", "_blank");
            wnd.document.write(metadata)
        """
            
    elif widget == 'stats':
        return """
            var term_to_stats = term_to_stats
            var term = multi_select.value[0]
            var stats_description = term_to_stats[term]
            var wnd = window.open("about:blank", "", "_blank");
            wnd.document.write(stats_description)
        """

    elif widget == 'class':
        return """
            var term_to_class = term_to_class;
            var multi_select = multi_select;
            var class_hex = class_hex;
            var active = cb_obj.active;

            if (active.length == 1) {
                class_hex.visible = true
            } else {
                class_hex.visible = false
            }
        """
            
    return ""


def plot(tables, output_filename, xmin, xmax, ymin, ymax, class_id):
    """
    Constructs the main Bokeh layout and saves it to an HTML file.
    """
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    file_name = f'figures/{output_filename}.html'
    output_file(file_name)
    
    BLUR_MAX = 4
    BLUR_STEP_SIZE = 0.25
    SATURATION_MAX = 5
    SATURATION_STEP_SIZE = 0.25
    SIZE_HEXAGONS = 10
    
    orientation = 'flattop'
    ratio = ((ymax - ymin) / (xmax - xmin))
    size = SIZE_HEXAGONS / ratio
    hexagon_height = sqrt(3) * size * ratio

    p = figure(x_range=[xmin, xmax], y_range=[ymin - (hexagon_height/2), ymax],
               tools="wheel_zoom,reset,save", background_fill_color='#440154')

    p.grid.visible = False
    p.xaxis.axis_label = "log(P)"
    p.yaxis.axis_label = "Molecular Weight (Da)"

    term_to_source = {}
    term_to_class = {}
    term_to_drug_sources = {}
    term_to_metadata = {}
    term_to_stats = {}
    options = []
    
    has_drug_data = any('Drug_Like' in tables[t]['table'].columns for t in tables)

    for term in tables:
        logging.info(f'Processing query: {term}')
        options.append((term, term))
        table = tables[term]['table']

        source, title = create_data_source(table, term, size, ratio, orientation, BLUR_MAX, BLUR_STEP_SIZE)
        source_class = create_class_source(table, size, ratio, orientation, class_id)
        
        drug_sources = {}
        if 'Drug_Like' in table.columns:
            drug_sources['Drug_Like'] = create_drug_like_source(table, size, ratio, orientation, 'Drug_Like')
        
        term_to_source[term] = {'source': source, 'title': title}
        term_to_class[term] = {'source': source_class, 'show_class': True}
        term_to_drug_sources[term] = {'source': drug_sources.get('Drug_Like', ColumnDataSource())}
        term_to_stats[term] = create_stats_description(table)
        term_to_metadata[term] = return_html(tables[term]['metadata'])

    default_term = list(tables.keys())[0]
    default_source, default_title = create_data_source(tables[default_term]['table'], default_term, size, ratio, orientation, BLUR_MAX, BLUR_STEP_SIZE)
    p.title.text = default_title

    mapper = linear_cmap('Count', 'Viridis256', 0, max(default_source.data.get('Count', [0])))
    hex_renderer = p.hex_tile(q="q", r="r", size=size, line_color=None, source=default_source, aspect_scale=ratio, orientation=orientation, fill_color=mapper)

    class_hex = None
    if class_id:
        source_class_def = create_class_source(tables[default_term]['table'], size, ratio, orientation, class_id)
        class_hex = p.hex_tile(q='q', r="r", size=size, line_color=None, source=source_class_def, aspect_scale=ratio, orientation=orientation, fill_color='#ff007f')
        class_hex.visible = False

    drug_hex = None
    if has_drug_data:
        source_drug_def = create_drug_like_source(tables[default_term]['table'], size, ratio, orientation, 'Drug_Like')
        drug_hex = p.hex_tile(q='q', r="r", size=size, line_color='#00ff00', line_width=2, source=source_drug_def, aspect_scale=ratio, orientation=orientation, fill_color='#00ff00', fill_alpha=0.3)
        drug_hex.visible = False

    hover = HoverTool(renderers=[hex_renderer], tooltips=return_JS_code('tooltips'), callback=CustomJS(code=return_JS_code('hover')), show_arrow=False)
    p.add_tools(hover)

    slider1 = Slider(start=1, end=SATURATION_MAX, value=1, step=SATURATION_STEP_SIZE, title="Saturation", width=100)
    slider2 = Slider(start=0, end=BLUR_MAX, value=0, step=BLUR_STEP_SIZE, title="Blur", width=100)
    multi_select = MultiSelect(title=output_filename, value=[default_term], options=options, width=100, height=200)
    checkbox = CheckboxGroup(labels=["TFIDF"], active=[])
    rbg = RadioGroup(labels=["Viridis256", "Greys256"], active=0)
    btn_meta = Button(label="Metadata", button_type="default", width=100)
    btn_stats = Button(label="Statistics", button_type="default", width=100)
    
    widgets = [slider1, slider2, checkbox, rbg]
    
    if has_drug_data:
        cb_drug = CheckboxGroup(labels=["Highlight Drug-like"], active=[])
        widgets.append(cb_drug)
        cb_drug.js_on_change('active', CustomJS(args={'source_drug': source_drug_def, 'term_to_drug_sources': term_to_drug_sources, 'multi_select': multi_select, 'drug_hex': drug_hex}, code=return_JS_code('drug_likeness')))

    if class_id:
        cb_class = CheckboxGroup(labels=[f"Show {class_id}"], active=[])
        widgets.append(cb_class)
        cb_class.js_on_change('active', CustomJS(args={'multi_select': multi_select, 'term_to_class': term_to_class, 'class_hex': class_hex}, code=return_JS_code('class')))

    widgets.extend([btn_meta, btn_stats])

    slider1.js_on_change('value', CustomJS(args={'source': default_source, 'mapper': mapper, 'slider2': slider2, 'checkbox': checkbox}, code=return_JS_code('slider1')))
    slider2.js_on_change('value', CustomJS(args={'source': default_source, 'mapper': mapper, 'slider1': slider1, 'checkbox': checkbox}, code=return_JS_code('slider2')))
    
    ms_args = {'source': default_source, 'term_to_source': term_to_source, 'slider1': slider1, 'slider2': slider2, 'checkbox': checkbox, 'p': p, 'mapper': mapper}
    if class_id:
        ms_args.update({'source_class': source_class_def, 'term_to_class': term_to_class, 'class_hex': class_hex, 'checkbox_class': cb_class})
    multi_select.js_on_change("value", CustomJS(args=ms_args, code=return_JS_code('multi_select_class' if class_id else 'multi_select')))
    
    checkbox.js_on_change('active', CustomJS(args={'source': default_source, 'slider2': slider2, 'mapper': mapper, 'slider1': slider1}, code=return_JS_code('checkbox')))
    rbg.js_on_change('active', CustomJS(args={'p': p, 'multi_select': multi_select, 'mapper': mapper, 'term_to_class': term_to_class, 'Viridis256': linear_cmap('Count', 'Viridis256', 0, 1)['transform'].palette, 'Greys256': linear_cmap('Count', 'Greys256', 0, 1)['transform'].palette}, code=return_JS_code('rbg')))
    
    btn_meta.js_on_event(events.ButtonClick, CustomJS(args={'term_to_metadata': term_to_metadata, 'multi_select': multi_select}, code=return_JS_code('button')))
    btn_stats.js_on_event(events.ButtonClick, CustomJS(args={'term_to_stats': term_to_stats, 'multi_select': multi_select}, code=return_JS_code('stats')))

    from bokeh.layouts import column, row
    layout = row(multi_select, p, column(*widgets))
    show(layout)


def parser():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive Chemical Space Visualization')
    parser.add_argument('-i', required=True, dest='input_folder', help='Input folder with table files (e.g. data/processed)')
    parser.add_argument('-o', default='Interactive_Chemical_Space', dest='output_filename', help='Output HTML filename')
    parser.add_argument('-xmin', type=float, default=-5, help='X axis minimum (logP)')
    parser.add_argument('-xmax', type=float, default=10, help='X axis maximum (logP)')
    parser.add_argument('-ymax', type=float, default=1600, help='Y axis maximum (Mass)')
    parser.add_argument('-class', dest='class_id', help='ChEBI Class ID to highlight')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parser()
    start_time = datetime.now()
    
    logging.info("Starting Interactive Visualization Pipeline...")
    
    required_dirs = ['figures', 'data/processed', 'data/metadata']
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(args.input_folder):
        logging.critical(f"Input folder '{args.input_folder}' not found.")
        return

    files = get_files(args.input_folder)
    if not files:
        logging.error("No data files found to process.")
        print("Error: No data files found. Please run data_processing.py first.")
        return

    logging.info(f"Processing {len(files)} files.")
    tables = get_tables(files)
    
    plot(tables, args.output_filename, args.xmin, args.xmax, 0, args.ymax, args.class_id)
    
    duration = datetime.now() - start_time
    logging.info(f"Visualization completed key generated in {duration}.")
    print(f"Interactive plot generation successful: figures/{args.output_filename}.html")


if __name__ == '__main__':
    main()