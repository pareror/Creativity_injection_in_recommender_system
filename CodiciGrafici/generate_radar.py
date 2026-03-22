import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurazione
EXCEL_DIR = '/home/pagano/Tirocinio/excel_novelty'
OUTPUT_DIR = '/home/pagano/Tirocinio/grafici_tradeoff_analysis'

# File specifici da processare con descrizione per il titolo
FILES_TO_PROCESS = {
    "Amazon Reranking bipolare alpha 0.1.xlsx": "Amazon (Bipolar Alpha=0.1)",
    "Amazon_creativity 03_03_03.xlsx": "Amazon (Creativity 0.33)",
    "Movielens Reranking bipolare alpha 0.1.xlsx": "Movielens (Bipolar Alpha=0.1)",
    "Movielens_creativity 03_03_03.xlsx": "Movielens (Creativity 0.33)"
}

# Modelli da escludere
EXCLUDE_LIST = ['Pop', 'multivae', 'itemknn', 'MultiVAE', 'ItemKNN']

def load_metric(file_path, sheet_name, keyword, is_reranked_k100=True):
    """Carica i dati di una metrica specifica (solo Reranked)."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        
        # Cerchiamo la colonna Reranked
        # Deve contenere 'keyword' e 'rerank'
        # Se is_reranked_k100 è True, deve anche contenere 'k100' (comune per molte metriche reranked)
        # Ma per alcune metriche potrebbe esserci solo 'reranked' senza 'k100', quindi controlliamo.
        
        col_rerank = None
        cols = df.columns
        
        # Tentativo 1: Standard name check
        if is_reranked_k100:
             col_rerank = next((c for c in cols if keyword.lower() in c.lower() and 'rerank' in c.lower() and 'k100' in c.lower() and 'delta' not in c.lower()), None)
        else:
             col_rerank = next((c for c in cols if keyword.lower() in c.lower() and 'rerank' in c.lower() and 'delta' not in c.lower()), None)
             
        # Fallback: se non trova con k100, prova senza
        if not col_rerank and is_reranked_k100:
             col_rerank = next((c for c in cols if keyword.lower() in c.lower() and 'rerank' in c.lower() and 'delta' not in c.lower()), None)

        if col_rerank:
            return df[['Model', col_rerank]].rename(columns={col_rerank: 'Value'})
        else:
            print(f"  Warning: Colonna Reranked non trovata per {sheet_name} (kw: {keyword}) in {os.path.basename(file_path)}")
            # Debug: stampa colonne
            # print(f"    Colonne disponibili: {list(cols)}")
            return None
    except Exception as e:
        print(f"  Errore caricamento {sheet_name}: {e}")
        return None

def load_unexpectedness(file_path):
    """Caricamento specifico per Unexpectedness che ha sheet e nomi un po' diversi."""
    try:
        df = pd.read_excel(file_path, sheet_name='Unexpectedness', engine='openpyxl')
        # Cerca colonna reranked
        col_rerank = next((c for c in df.columns if 'reranked' in c.lower() and '10' in c.lower() and 'k100' in c.lower()), None)
        
        # Fallback
        if not col_rerank:
             col_rerank = next((c for c in df.columns if 'reranked' in c.lower() and '10' in c.lower()), None)
             
        if col_rerank:
            return df[['Model', col_rerank]].rename(columns={col_rerank: 'Value'})
        else:
            print(f"  Warning: Colonna Unexpectedness Reranked non trovata in {os.path.basename(file_path)}")
            return None
    except Exception as e:
        print(f"  Errore Unexpectedness: {e}")
        return None

def load_all_metrics(file_path):
    """Carica NDCG, Novelty, Unexpectedness."""
    data = {}
    
    # 1. Relevance: NDCG@10
    df_ndcg = load_metric(file_path, 'ndcg', 'ndcg@10')
    if df_ndcg is not None:
        data['Relevance'] = df_ndcg
        
    # 2. Novelty: Novelty@10
    df_nov = load_metric(file_path, 'Novelty', 'novelty@10', is_reranked_k100=False) # Novelty sheet often simpler
    # Se fallisce, riprova con default logic
    if df_nov is None:
        df_nov = load_metric(file_path, 'Novelty', 'novelty@10', is_reranked_k100=True)
        
    if df_nov is not None:
        data['Novelty'] = df_nov
        
    # 3. Unexpectedness: Unexpectedness@10
    df_unexp = load_unexpectedness(file_path)
    if df_unexp is not None:
        data['Unexpectedness'] = df_unexp
        
    return data

def normalize_series(series):
    """Normalizza una serie tra 0 e 1 (Min-Max scaling)."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series.apply(lambda x: 1.0 if x > 0 else 0.0) # Avoid div by zero
    return (series - min_val) / (max_val - min_val)

def plot_radar_chart(dataset_name, merged_df, output_dir):
    """Genera un Radar Chart (Spider Plot) per confrontare le metriche."""
    
    # Filtra modelli
    merged_df = merged_df[~merged_df['Model'].isin(EXCLUDE_LIST)].copy()
    
    if merged_df.empty:
        print(f"  Nessun dato da plottare per {dataset_name}")
        return

    # Normalizzazione (Min-Max per avere tutto tra 0 e 1)
    # Per Relevance (NDCG), Novelty, Unexpectedness
    # Usiamo Min-Max scaling relativo al dataset corrente per evidenziare le differenze
    for col in ['Relevance', 'Novelty', 'Unexpectedness']:
        min_val = merged_df[col].min()
        max_val = merged_df[col].max()
        if max_val - min_val > 0:
            merged_df[f'{col}_Norm'] = (merged_df[col] - min_val) / (max_val - min_val)
        else:
             merged_df[f'{col}_Norm'] = 1.0 # Se tutti uguali, max value
             
    # Categorie (Assi)
    categories = ['Relevance', 'Novelty', 'Unexpectedness']
    N = len(categories)
    
    # Angoli per gli assi
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Chiudere il cerchio
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Offset rotazione per avere il primo asse in alto
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Disegna assi e etichette
    plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
    
    # Disegna y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.1)
    
    # Colori distinti ad alto contrasto
    # Assicuriamo che CFKG e LightGCN abbiano colori ben diversi
    custom_colors = [
        '#e6194b', # Red
        '#3cb44b', # Green
        '#ffe119', # Yellow
        '#4363d8', # Blue
        '#f58231', # Orange
        '#911eb4', # Purple
        '#46f0f0', # Cyan
        '#f032e6', # Magenta
        '#bcf60c', # Lime
        '#fabebe', # Pink
        '#008080', # Teal
        '#e6beff', # Lavender
        '#9a6324', # Brown
        '#fffac8', # Beige
        '#800000', # Maroon
        '#aaffc3', # Mint
        '#808000', # Olive
        '#ffd8b1', # Apricot
        '#000075', # Navy
        '#808080', # Gray
    ]
    
    # Se ci sono più modelli dei colori, facciamo wrap-around ma non dovrebbe succedere con <= 9 modelli
    
    for idx, row in merged_df.iterrows():
        model_name = row['Model']
        values = [row['Relevance_Norm'], row['Novelty_Norm'], row['Unexpectedness_Norm']]
        values += values[:1] # Chiudere il poligono
        
        color = custom_colors[idx % len(custom_colors)]
        
        # Line width leggermente ridotte per evitare 'pasticcio' se troppe linee
        # Aggiungiamo un marker per aiutare a distinguere i punti
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=color, marker='o', markersize=4)
        ax.fill(angles, values, alpha=0.05, color=color) # Alpha molto basso per il fill per non coprire troppo
    
    # Titolo e Legenda
    plt.title(f'Trade-off Analysis (Normalized)\nDataset: {dataset_name}', size=16, color='black', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    # Save
    save_dir = os.path.join(output_dir, 'radar')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"{dataset_name}_Radar_Chart.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafico salvato: {filepath}")

def main():
    if not os.path.exists(EXCEL_DIR):
        print(f"Directory non trovata: {EXCEL_DIR}")
        return
        
    for filename, dataset_name in FILES_TO_PROCESS.items():
        file_path = os.path.join(EXCEL_DIR, filename)
        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            continue
            
        print(f"Processando {dataset_name} ({filename})...")
        metrics_data = load_all_metrics(file_path)
        
        # Merge dei dataframe
        if 'Relevance' in metrics_data and 'Novelty' in metrics_data and 'Unexpectedness' in metrics_data:
            df_final = metrics_data['Relevance']
            df_final = pd.merge(df_final, metrics_data['Novelty'], on='Model', suffixes=('', '_nov'))
            df_final = df_final.rename(columns={'Value': 'Relevance', 'Value_nov': 'Novelty'})
            
            df_final = pd.merge(df_final, metrics_data['Unexpectedness'], on='Model', suffixes=('', '_unexp'))
            df_final = df_final.rename(columns={'Value': 'Unexpectedness'})
            
            # Plot Radar
            plot_radar_chart(dataset_name, df_final, OUTPUT_DIR)
        else:
            print(f"  Dati mancanti per {dataset_name}: {metrics_data.keys()}")

if __name__ == "__main__":
    main()
