"""
Script per generare grafici trend: Metrica vs TopK (5, 5_reranked, 10, 10_reranked)

- Asse X: TopK configuration (@5, @5_reranked, @10, @10_reranked)
- Asse Y: Valore della metrica
- Ogni modello ha una linea colorata diversa

Metriche generate: NDCG, Recall, AvgPop, Gini, Unexpectedness, Serendipity_Yan, Serendipity_Ge

I grafici vengono salvati in: grafici_trend/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurazione
EXCEL_DIR = os.path.join(os.path.dirname(__file__), "excel")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "grafici_trend")

# Mapping file Excel -> (dataset, metodo, is_creativity)
EXCEL_FILES = {
    "Movielens Reranking bipolare alpha 0.1.xlsx": ("Movielens", "Bipolar Reranking", False),
    "Amazon Reranking bipolare alpha 0.1.xlsx": ("Amazon", "Bipolar Reranking", False),
    "Movielens_creativity 03_03_03.xlsx": ("Movielens", "Creativity Score", True),
    "Amazon_creativity 03_03_03.xlsx": ("Amazon", "Creativity Score", True),
}

# Configurazione metriche con mapping colonne per file bipolare
# Formato: (sheet_name, display_name, col_5, col_5_rerank, col_10, col_10_rerank)
METRICS_BIPOLARE = {
    "ndcg": ("ndcg", "NDCG", "ndcg@5", "ndcg@5_reranked", "ndcg@10", "ndcg@10_reranked"),
    "recall": ("recall", "Recall", "recall@5", "recall@5_reranked", "recall@10", "recall@10_reranked"),
    "avgpop": ("averagepopularity", "Average Popularity", "averagepopularity@5", "averagepopularity@5_reranked", "averagepopularity@10", "averagepopularity@10_reranked"),
    "gini": ("giniindex", "Gini Index", "giniindex@5", "giniindex@5_reranked", "giniindex@10", "giniindex@10_reranked"),
    "unexp": ("Unexpectedness", "Unexpectedness", "Unexpectedness_Original@5", "Unexpectedness_Reranked@5", "Unexpectedness_Original@10", "Unexpectedness_Reranked@10"),
    "serendipity_yan": ("Serendipity_Yan", "Serendipity (Yan)", "Serendipity_Yan_Original@5", "Serendipity_Yan_Reranked@5", "Serendipity_Yan_Original@10", "Serendipity_Yan_Reranked@10"),
    "serendipity_ge": ("Serendipity_Ge", "Serendipity (Ge)", "Serendipity_Ge_Original@5", "Serendipity_Ge_Reranked@5", "Serendipity_Ge_Original@10", "Serendipity_Ge_Reranked@10"),
}

# Colori distintivi per ogni modello
MODEL_COLORS = {
    'ItemKNN': '#1f77b4',    # Blu
    'DMF': '#ff7f0e',        # Arancione
    'Pop': '#2ca02c',        # Verde
    'MultiVAE': '#d62728',   # Rosso
    'KGCN': '#9467bd',       # Viola
    'LightGCN': '#8c564b',   # Marrone
    'CKE': '#e377c2',        # Rosa
    'CFKG': '#7f7f7f',       # Grigio
    'KGNNLS': '#bcbd22',     # Giallo-verde
    'MKR': '#17becf',        # Ciano
    'BPR': '#ff9896',        # Rosa chiaro
    'ENMF': '#aec7e8',       # Azzurro chiaro
    'MULTIDAE': '#ffbb78',   # Arancione chiaro
}


def load_metric_data_bipolare(excel_path, metric_key):
    """Carica i dati di una metrica dal foglio Excel (file bipolare)."""
    config = METRICS_BIPOLARE[metric_key]
    sheet_name, display_name, col_5, col_5_rerank, col_10, col_10_rerank = config
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"⚠️ Errore lettura foglio {sheet_name}: {e}")
        return None, None
    
    # Trova le colonne (case-insensitive matching)
    cols = df.columns.tolist()
    cols_lower = [c.lower() for c in cols]
    
    def find_col(target):
        target_lower = target.lower()
        for i, c in enumerate(cols_lower):
            if c == target_lower:
                return cols[i]
        return None
    
    col_5_found = find_col(col_5)
    col_5_rerank_found = find_col(col_5_rerank)
    col_10_found = find_col(col_10)
    col_10_rerank_found = find_col(col_10_rerank)
    
    if not all([col_5_found, col_5_rerank_found, col_10_found, col_10_rerank_found]):
        print(f"⚠️ Colonne mancanti per {metric_key}: {cols}")
        return None, None
    
    data = []
    for _, row in df.iterrows():
        model = row['Model']
        data.append({
            'Model': model,
            '@5': row[col_5_found],
            '@5_reranked': row[col_5_rerank_found],
            '@10': row[col_10_found],
            '@10_reranked': row[col_10_rerank_found],
        })
    
    return pd.DataFrame(data), display_name


def load_metric_data_creativity(excel_path, metric_key):
    """Carica i dati di una metrica dal foglio Excel (file creativity)."""
    config = METRICS_BIPOLARE[metric_key]
    sheet_name, display_name = config[0], config[1]
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"⚠️ Errore lettura foglio {sheet_name}: {e}")
        return None, None
    
    cols = df.columns.tolist()
    
    # Per creativity, le colonne reranked hanno _K100
    def find_creativity_cols():
        col_5 = None
        col_5_rerank = None
        col_10 = None
        col_10_rerank = None
        
        for col in cols:
            col_lower = col.lower()
            
            # Cerca colonne @5
            if '@5' in col_lower and 'delta' not in col_lower:
                if 'rerank' in col_lower and 'k100' in col_lower:
                    col_5_rerank = col
                elif 'rerank' not in col_lower and 'k50' not in col_lower and 'k100' not in col_lower:
                    if 'original' in col_lower or col_lower.endswith('@5'):
                        col_5 = col
            
            # Cerca colonne @10
            if '@10' in col_lower and 'delta' not in col_lower:
                if 'rerank' in col_lower and 'k100' in col_lower:
                    col_10_rerank = col
                elif 'rerank' not in col_lower and 'k50' not in col_lower and 'k100' not in col_lower:
                    if 'original' in col_lower or col_lower.endswith('@10'):
                        col_10 = col
        
        return col_5, col_5_rerank, col_10, col_10_rerank
    
    col_5, col_5_rerank, col_10, col_10_rerank = find_creativity_cols()
    
    if not all([col_5, col_5_rerank, col_10, col_10_rerank]):
        print(f"⚠️ Colonne creativity mancanti per {metric_key}: trovate {col_5}, {col_5_rerank}, {col_10}, {col_10_rerank} in {cols}")
        return None, None
    
    data = []
    for _, row in df.iterrows():
        model = row['Model']
        data.append({
            'Model': model,
            '@5': row[col_5],
            '@5_reranked': row[col_5_rerank],
            '@10': row[col_10],
            '@10_reranked': row[col_10_rerank],
        })
    
    return pd.DataFrame(data), display_name


def create_trend_plot(df, metric_name, dataset, method, output_dir, k, use_log_scale=False):
    """Crea e salva il grafico trend per un valore specifico di K."""
    
    if df is None or df.empty:
        print(f"⚠️ Nessun dato per {metric_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configurazione asse X per il K specifico
    x_labels = [f'@{k}', f'@{k}\nreranked']
    x_positions = [0, 1]
    
    col_orig = f'@{k}'
    col_rerank = f'@{k}_reranked'
    
    # Filtra modelli da escludere
    exclude_list = ['Pop', 'multivae', 'itemknn', 'MultiVAE', 'ItemKNN']
    df = df[~df['Model'].isin(exclude_list)]
    
    if df.empty:
        print(f"⚠️ Tutti i modelli filtrati per {metric_name}")
        return

    # Plot linea per ogni modello
    for _, row in df.iterrows():
        model = row['Model']
        y_values = [row[col_orig], row[col_rerank]]
        
        color = MODEL_COLORS.get(model, '#333333')
        
        ax.plot(x_positions, y_values, 'o-', color=color, linewidth=2, 
                markersize=8, label=model, markeredgecolor='white', markeredgewidth=1)
    
    # Configurazione grafico
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel('TopK Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{metric_name} @{k}', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset} - {method}\n{metric_name} Trend @{k}', fontsize=15, fontweight='bold')
    
    if use_log_scale:
        ax.set_yscale('log')
    
    # Legenda fuori dal grafico
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, 1.5)  # Restringi i limiti per 2 punti
    
    # Sfondo bianco
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome file con indicazione del K
    safe_metric = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    safe_dataset = dataset.lower().replace(' ', '_')
    safe_method = method.lower().replace(' ', '_')
    filename = os.path.join(output_dir, f'trend_{safe_metric}_{safe_dataset}_{safe_method}_k{k}.png')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Salvato: {filename}")


def main():
    print("=" * 70)
    print("GENERATORE GRAFICI TREND: Metrica vs TopK Configuration")
    print("=" * 70)
    
    total_plots = 0
    
    for excel_filename, (dataset, method, is_creativity) in EXCEL_FILES.items():
        excel_path = os.path.join(EXCEL_DIR, excel_filename)
        
        if not os.path.exists(excel_path):
            print(f"⚠️ File non trovato: {excel_path}")
            continue
        
        print(f"\n📂 Processing: {excel_filename}")
        print(f"   Dataset: {dataset}, Metodo: {method}")
        
        # Genera grafici per ogni metrica
        for metric_key in METRICS_BIPOLARE.keys():
            try:
                if is_creativity:
                    df, metric_name = load_metric_data_creativity(excel_path, metric_key)
                else:
                    df, metric_name = load_metric_data_bipolare(excel_path, metric_key)
                
                if df is None:
                    continue
                
                # Genera grafici separati per @5 e @10
                for k in [5, 10]:
                    create_trend_plot(df, metric_name, dataset, method, OUTPUT_DIR, k=k)
                total_plots += 2
                
            except Exception as e:
                print(f"❌ Errore per {metric_key}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print(f"🎉 Completato! Generati {total_plots} grafici in {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
