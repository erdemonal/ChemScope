"""
chemometrics_analysis.py

This module implements the chemometric analysis pipeline for the ChemScope project.
It performs data consolidation, multivariate analysis (PCA), clustering (K-Means),
statistical hypothesis testing (ANOVA, T-tests), and bias quantification.

Dependencies: pandas, matplotlib, seaborn, scikit-learn, scipy
"""

import logging
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import f_oneway


os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/chemometrics_analysis.log', mode='w')
    ]
)

sns.set(style="whitegrid", font_scale=1.2)


def load_all_data() -> pd.DataFrame:
    """
    Consolidates data tables from different analytical techniques into a single DataFrame.
    
    Returns:
        pd.DataFrame: A consolidated DataFrame containing compounds from all techniques.
    
    Raises:
        FileNotFoundError: If no data files are found in the 'data/processed' directory.
    """
    all_files = glob.glob("data/processed/*_Binding_Dataset.tsv")
    if not all_files:
        all_files = glob.glob("*_Binding_Dataset.tsv")
         
    df_list = []
    
    logging.info(f"Found {len(all_files)} data files.")
    
    for f in all_files:
        term_name = os.path.basename(f).split('_Binding_Dataset')[0]
        technique = term_name.split('-')[0] if '-' in term_name else term_name
        
        try:
            temp_df = pd.read_csv(f, sep='\t', index_col=0)
            temp_df['Technique'] = technique
            df_list.append(temp_df)
            logging.info(f"Loaded {term_name}: {len(temp_df)} compounds")
        except Exception as e:
            logging.error(f"Error loading {f}: {e}")
        
    if not df_list:
        raise FileNotFoundError("No table files found. Please conduct data processing first.")
        
    return pd.concat(df_list)


def perform_pca_and_loadings(df: pd.DataFrame):
    """
    Performs Principal Component Analysis to visualize the chemical space distribution
    and computes feature importance (loadings).

    Args:
        df (pd.DataFrame): The input dataset containing chemical properties and roles.

    Returns:
        tuple: (pca_df, explained_variance, loadings)
            pca_df (pd.DataFrame): Transformed data with PC coordinates.
            explained_variance (array): Explained variance ratio for PC1 and PC2.
            loadings (pd.DataFrame): Feature contributions to principal components.
    """
    logging.info("Starting PCA Analysis.")
    
    if 'Role' in df.columns:
        df_pca = df[df['Role'] == 'Ligand'].copy()
        logging.info(f"Filtering for 'Ligand' role: {len(df_pca)} compounds remain.")
    else:
        df_pca = df.copy()

    features = ['Mass', 'logP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
    available_features = [f for f in features if f in df_pca.columns]
    
    if len(available_features) < 2:
        logging.warning("Insufficient features available for PCA.")
        return None, None, None

    df_pca = df_pca.dropna(subset=available_features)
    logging.info(f"Data points available after removing NaNs: {len(df_pca)}")
    
    x = df_pca.loc[:, available_features].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_scaled)
    
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['Technique'] = df_pca['Technique'].values
    if 'Names' in df_pca.columns:
        pca_df['Names'] = df_pca['Names'].values
    
    explained_variance = pca.explained_variance_ratio_
    logging.info(f"Explained Variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}")
    
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=available_features
    )
    logging.info("PCA Loadings computed.")
    
    return pca_df, explained_variance, loadings


def plot_pca_results(pca_df: pd.DataFrame, explained_variance, loadings: pd.DataFrame):
    """
    Generates and saves PCA scatter plot and loadings biplot.

    Args:
        pca_df (pd.DataFrame): PCA coordinates and metadata.
        explained_variance (array): Explained variance ratios.
        loadings (pd.DataFrame): Feature loadings.
    """
    os.makedirs('figures/chemometrics', exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Technique',
        data=pca_df,
        palette='viridis',
        alpha=0.6,
        s=60,
        style='Technique'
    )
    
    plt.title(f'PCA of Chemical Space (n={len(pca_df)})', fontsize=16, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
    plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/chemometrics/pca_techniques_space.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(loadings.index):
        plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], 
                 color='r', alpha=0.5, width=0.01, head_width=0.05)
        plt.text(loadings.iloc[i, 0]*1.15, loadings.iloc[i, 1]*1.15, 
                feature, color='darkred', ha='center', va='center', fontweight='bold')
        
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    
    circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_artist(circle)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title('Feature Contributions (PCA Loadings)', fontsize=16)
    plt.xlabel('Correlation with PC1', fontsize=14)
    plt.ylabel('Correlation with PC2', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/chemometrics/pca_loadings_vectors.png', dpi=300)
    plt.close()
    
    logging.info("PCA plots saved to 'figures/chemometrics/'")


def perform_clustering_analysis(df: pd.DataFrame, n_clusters: int = 5):
    """
    Performs K-Means clustering to identify mathematical groupings within the chemical space.
    
    Args:
        df (pd.DataFrame): The input data.
        n_clusters (int): The number of clusters to form. Defaults to 5.
    """
    logging.info("Starting K-Means Clustering Analysis.")
    
    if 'Role' in df.columns:
        df_clus = df[df['Role'] == 'Ligand'].copy()
    else:
        df_clus = df.copy()
        
    features = ['Mass', 'logP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
    available_features = [f for f in features if f in df_clus.columns]
    df_clus = df_clus.dropna(subset=available_features)
    
    x = StandardScaler().fit_transform(df_clus.loc[:, available_features].values)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clus['Cluster'] = kmeans.fit_predict(x)
    
    contingency = pd.crosstab(df_clus['Technique'], df_clus['Cluster'], normalize='index') * 100
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title('Technique Preference for Chemical Clusters (%)', fontsize=14)
    plt.ylabel('Analytical Technique')
    plt.xlabel('Automated Chemical Cluster (K-Means)')
    plt.tight_layout()
    plt.savefig('figures/chemometrics/cluster_technique_preference.png', dpi=300)
    plt.close()
    
    logging.info("Clustering heatmap saved.")
    
    cluster_stats = df_clus.groupby('Cluster')[available_features].mean()
    logging.info("Cluster Center Characteristics computed.")
    cluster_stats.to_csv('data/processed/stats_cluster_centers.csv')


def perform_statistical_tests(df: pd.DataFrame):
    """
    Performs inferential statistics (ANOVA) to assess significant differences between techniques.
    Also generates violin plots for distribution visualization.
    
    Args:
        df (pd.DataFrame): Consolidated data.
    """
    logging.info("Starting Statistical Hypothesis Testing.")
    
    if 'Role' in df.columns:
        df_stats = df[df['Role'] == 'Ligand'].copy()
    else:
        df_stats = df.copy()
        
    df_stats = df_stats.reset_index(drop=True)
        
    features = ['Mass', 'logP', 'TPSA']
    techniques = df_stats['Technique'].unique()
    
    results = []
    
    logging.info(f"Comparing techniques: {techniques}")
    
    for feature in features:
        if feature not in df_stats.columns:
            continue
        
        groups = [df_stats[df_stats['Technique'] == t][feature].dropna() for t in techniques]
        groups = [g for g in groups if len(g) > 2]
        
        if len(groups) < 2:
            continue
        
        f_stat, p_val = f_oneway(*groups)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        logging.info(f"{feature:10s} | ANOVA F={f_stat:6.2f} | p={p_val:.2e} {sig}")
        
        results.append({'Feature': feature, 'Test': 'ANOVA', 'Stat': f_stat, 'P-Value': p_val, 'Significance': sig})
        
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Technique', y=feature, hue='Technique', data=df_stats, palette="muted", inner="quartile", legend=False)
        plt.title(f'Distribution of {feature} across Techniques (p={p_val:.2e})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'figures/chemometrics/stats_violin_{feature}.png', dpi=300)
        plt.close()

    pd.DataFrame(results).to_csv('data/processed/stats_significance_tests.csv', index=False)
    logging.info("Statistical test results saved to 'data/processed/stats_significance_tests.csv'.")


def analyze_ubiquitous_chemicals(df: pd.DataFrame):
    """
    Quantifies bias by identifying 'ubiquitous' chemicals present across multiple analytical techniques.
    
    Args:
        df (pd.DataFrame): Consolidated data.
    """
    logging.info("Starting Bias Analysis (Ubiquitious Chemicals).")
    
    crosstab = pd.crosstab(df.index, df['Technique'])
    
    crosstab['Technique_Count'] = crosstab.gt(0).sum(axis=1)
    
    total_techniques = len(crosstab.columns) - 1
    ubiquitous = crosstab[crosstab['Technique_Count'] == total_techniques]
    
    logging.info(f"Total Unique Chemicals Analyzed: {len(crosstab)}")
    logging.info(f"Ubiquitous Chemicals (Present in ALL {total_techniques} techniques): {len(ubiquitous)}")
    
    if len(ubiquitous) > 0:
        names = df.groupby(level=0)['Names'].first()
        ubiquitous_names = ubiquitous.join(names)
        
        ubiquitous.to_csv('data/processed/bias_ubiquitous_chemicals.csv')
        logging.info("List of ubiquitous chemicals saved to 'data/processed/bias_ubiquitous_chemicals.csv'.")
    else:
        logging.info("No chemicals found present in all techniques.")


def analyze_non_ligands(df: pd.DataFrame):
    """
    Provides a quantitative breakdown of non-ligand entities (e.g., buffers, solvents) per technique.
    
    Args:
        df (pd.DataFrame): Consolidated data.
    """
    logging.info("Starting Non-Ligand Frequency Analysis.")
    
    if 'Role' not in df.columns:
        logging.warning("Role column missing. Skipping non-ligand analysis.")
        return

    non_ligands = df[df['Role'] != 'Ligand']
    
    if len(non_ligands) == 0:
        logging.info("No non-ligands found in dataset.")
        return
        
    logging.info(f"Total Non-Ligands identified: {len(non_ligands)}")
    
    stats = []
    
    for tech in df['Technique'].unique():
        tech_df = non_ligands[non_ligands['Technique'] == tech]
        if 'Names' in tech_df.columns:
            top_entities = tech_df['Names'].value_counts().head(10)
            
            for name, count in top_entities.items():
                role = tech_df[tech_df['Names'] == name]['Role'].iloc[0] if not tech_df[tech_df['Names'] == name].empty else 'Unknown'
                stats.append({
                    'Technique': tech,
                    'Entity': name,
                    'Count': count,
                    'Role': role
                })
    
    if stats:
        pd.DataFrame(stats).to_csv('data/processed/stats_top_non_ligands.csv', index=False)
        logging.info("Top non-ligand statistics saved to 'data/processed/stats_top_non_ligands.csv'.")


def main():
    """
    Main execution pipeline for Chemometric Analysis.
    """
    print("=== ChemScope Chemometrics Pipeline ===")
    logging.info("Pipeline started.")
    
    try:
        full_data = load_all_data()
        logging.info(f"Total consolidated entries: {len(full_data)}")
        
        if 'HBD' not in full_data.columns:
             logging.warning("RDKit properties (HBD, HBA) missing. Please check data processing step.")
    except Exception as e:
        logging.critical(f"Critical pipeline failure: {e}")
        return

    analyze_ubiquitous_chemicals(full_data)
    analyze_non_ligands(full_data)

    pca_df, explained_var, loadings = perform_pca_and_loadings(full_data)
    if pca_df is not None:
        plot_pca_results(pca_df, explained_var, loadings)
    
    perform_clustering_analysis(full_data)
    perform_statistical_tests(full_data)
    
    logging.info("Analysis completed successfully.")
    print("\n=== Analysis Complete ===")
    print("Outputs generated in:")
    print(" - data/processed/")
    print(" - figures/chemometrics/")


if __name__ == "__main__":
    main()
