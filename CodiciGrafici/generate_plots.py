"""
Script automatico per generare grafici Metrica vs Unexpectedness/Novelty.

- Legge i dati dai file Excel in /home/pagano/Tirocinio/excel/
- Genera grafici per ogni combinazione di:
  - Dataset (Movielens, Amazon)
  - Metodo reranking (Bipolare, Creativity Score)
  - Metrica Y (NDCG, Recall, Precision, GiniIndex)
  - Metrica X (Unexpectedness, Novelty)
- Asse X: Unexpectedness / Novelty
- Asse Y: Metrica scelta
- Blu: Pre-reranking (originale)
- Rosso: Post-reranking

I grafici vengono salvati in: grafici/{MetricaX}/

NOTA: Per i file creativity, usa le colonne _K100 con @10
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurazione
EXCEL_DIR = os.path.join(os.path.dirname(__file__), "excel_creativity")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "grafici")

# Metriche da plottare (nome sheet -> nome visualizzato)
Y_METRICS = {
    "ndcg": "NDCG@10",
    "recall": "Recall@10",
    "precision": "Precision@10",
    "giniindex": "GiniIndex@10",
}

# Metriche asse X
X_METRICS = ["Unexpectedness", "Novelty"]


def parse_filename(filename):
    """
    Estrae dataset, metodo e info da filename.
    Restituisce: (dataset, method_name, is_creativity)
    """
    filename_lower = filename.lower()
    
    # Dataset
    if "movielens" in filename_lower:
        dataset = "Movielens"
    elif "amazon" in filename_lower:
        dataset = "Amazon"
    else:
        dataset = "Unknown"
        
    # Method & Creativity
    if "creativity" in filename_lower:
        is_creativity = True
        method = "Creativity Score"
        # Cerca varianti (pesi o alpha) per distinguere nel titolo
        if "03_03_03" in filename_lower:
            method += " (weights 0.3)"
        elif "25_25_50" in filename_lower:
            method += " (weights 25-25-50)"
        elif "25_50_25" in filename_lower:
            method += " (weights 25-50-25)"
        elif "50_25_25" in filename_lower:
            method += " (weights 50-25-25)"
    elif "bipolare" in filename_lower:
        is_creativity = False
        method = "Bipolar Reranking"
        # Cerca alpha
        if "alpha 0.1" in filename_lower:
            method += " (alpha 0.1)"
        elif "alpha 0.3" in filename_lower:
            method += " (alpha 0.3)"
        elif "alpha 0.5" in filename_lower:
            method += " (alpha 0.5)"
    else:
        is_creativity = False
        method = "Unknown Method"
        
    return dataset, method, is_creativity


def load_metric_data(excel_path, sheet_name, is_creativity):
    """Carica i dati di una metrica dal foglio Excel (Asse Y)."""
    try:
        # engine='openpyxl' è più sicuro per xlsx moderni
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        # print(f"⚠️ Errore caricamento sheet {sheet_name}: {e}")
        return None
    
    cols = df.columns.tolist()
    original_col = None
    reranked_col = None
    
    for col in cols:
        col_lower = str(col).lower()
        
        # Cerca la colonna originale (es. ndcg@10)
        # Deve contenere @10 e non 'rerank' o 'delta'
        if '@10' in col_lower and 'delta' not in col_lower and 'rerank' not in col_lower:
            original_col = col
        
        # Cerca la colonna reranked
        if is_creativity:
            # Per creativity: cerca _reranked_K100 con @10
            if '@10_reranked_k100' in col_lower and 'delta' not in col_lower:
                reranked_col = col
        else:
            # Per bipolare: cerca _reranked con @10
            if '@10_rerank' in col_lower and 'delta' not in col_lower:
                reranked_col = col
    
    # Trova la colonna Model (case-insensitive)
    model_col = next((c for c in cols if str(c).lower() == 'model'), None)
    if model_col is None:
        return None

    if original_col is None or reranked_col is None:
        return None
    
    return df[[model_col, original_col, reranked_col]].rename(columns={
        model_col: 'Model',
        original_col: 'original',
        reranked_col: 'reranked'
    })


def load_x_axis_data(excel_path, metric_name, is_creativity):
    """Carica i dati per l'asse X (Unexpectedness o Novelty)."""
    try:
        df = pd.read_excel(excel_path, sheet_name=metric_name, engine='openpyxl')
    except:
        # Silenzioso se il foglio non esiste (es. Novelty in alcuni file vecchi)
        return None
    
    cols = df.columns.tolist()
    original_col = None
    reranked_col = None
    
    for col in cols:
        col_str = str(col).lower()
        metric_lower = metric_name.lower()
        
        # Cerca Colonna originale
        # 1. Priorità: Contiene 'original' e '10'
        if 'original' in col_str and '10' in col_str:
            original_col = col
        # 2. Fallback: Contiene mome metrica e '10', ma NON 'rerank' o 'delta'
        elif metric_lower in col_str and '10' in col_str and 'rerank' not in col_str and 'delta' not in col_str:
            if original_col is None:
                original_col = col
        
        # Colonna reranked
        if is_creativity:
            # Per creativity: {Metric}_Reranked@10_K100
            if 'reranked' in col_str and '10' in col_str and 'k100' in col_str:
                reranked_col = col
        else:
            # Per bipolare: {Metric}_Reranked@10 (senza K)
            if 'reranked' in col_str and '10' in col_str and 'k50' not in col_str and 'k100' not in col_str:
                reranked_col = col
    
    # Trova la colonna Model (case-insensitive)
    model_col = next((c for c in cols if str(c).lower() == 'model'), None)
    if model_col is None:
        return None
    
    if original_col is None or reranked_col is None:
        return None
    
    return df[[model_col, original_col, reranked_col]].rename(columns={
        model_col: 'Model',
        original_col: 'x_original',
        reranked_col: 'x_reranked'
    })


def create_plot(y_metric_name, y_data, x_metric_name, x_data, dataset, method, output_dir):
    """Crea e salva il grafico."""
    
    # Merge dei dati
    # Usa suffix per evitare collisioni se le colonne hanno nomi uguali (non dovrebbe succedere col rename, ma sicurezza)
    df = pd.merge(y_data, x_data, on='Model', suffixes=('_y', '_x'))

    # Filtra modelli da escludere
    exclude_list = ['Pop', 'multivae', 'itemknn', 'MultiVAE', 'ItemKNN']
    df = df[~df['Model'].isin(exclude_list)]
    
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot punti originali (blu) e rerankati (rosso)
    for _, row in df.iterrows():
        model = row['Model']
        x_orig = row['x_original']
        y_orig = row['original']
        x_rerank = row['x_reranked']
        y_rerank = row['reranked']
        
        # Punto originale (blu)
        ax.scatter(x_orig, y_orig, c='blue', s=100, zorder=5, edgecolors='darkblue', linewidths=1)
        ax.annotate(model, (x_orig, y_orig), textcoords="offset points", 
                   xytext=(5, 5), fontsize=9, color='blue', fontweight='bold')
        
        # Punto rerankato (rosso)
        ax.scatter(x_rerank, y_rerank, c='red', s=100, zorder=5, edgecolors='darkred', linewidths=1)
        ax.annotate(model, (x_rerank, y_rerank), textcoords="offset points", 
                   xytext=(5, -12), fontsize=9, color='red', fontweight='bold')
        
        # Freccia di connessione
        ax.annotate('', xy=(x_rerank, y_rerank), xytext=(x_orig, y_orig),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))
    
    ax.set_xlabel(f'{x_metric_name}@10', fontsize=13)
    ax.set_ylabel(y_metric_name, fontsize=13)
    ax.set_title(f'{y_metric_name} vs {x_metric_name}\n{dataset} - {method}', fontsize=15, fontweight='bold')
    
    # Legenda
    ax.scatter([], [], c='blue', s=100, edgecolors='darkblue', label='Pre-reranking (Original)')
    ax.scatter([], [], c='red', s=100, edgecolors='darkred', label='Post-reranking')
    ax.legend(loc='best', fontsize=11)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Crea directory output specifica per la metrica X
    save_dir = os.path.join(output_dir, x_metric_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Nome file: pulizia stringhe
    safe_y = y_metric_name.lower().replace('@', '_').replace(' ', '_')
    safe_x = x_metric_name.lower().replace('@', '_').replace(' ', '_')
    safe_dataset = dataset.lower().replace(' ', '_')
    # Per il metodo, prendiamo solo la parte principale o tutto pulito
    safe_method = method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    filename = os.path.join(save_dir, f'{safe_y}_vs_{safe_x}_{safe_dataset}_{safe_method}.png')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Salvato: {filename}")


def main():
    print("=" * 70)
    print("GENERATORE AUTOMATICO GRAFICI: Metrica vs Unexpectedness/Novelty")
    print(f"Sorgente: {EXCEL_DIR}")
    print("=" * 70)
    
    if not os.path.exists(EXCEL_DIR):
        print(f"❌ Errore: Directory non trovata: {EXCEL_DIR}")
        return

    # Trova tutti i file xlsx
    files = [f for f in os.listdir(EXCEL_DIR) if f.endswith('.xlsx') and not f.startswith('~$')]
    files.sort()
    
    total_plots = 0
    
    for excel_filename in files:
        excel_path = os.path.join(EXCEL_DIR, excel_filename)
        
        # Parse del nome file
        dataset, method, is_creativity = parse_filename(excel_filename)
        
        print(f"\n📂 Processing: {excel_filename}")
        print(f"   ► Dataset: {dataset}, Metodo: {method}, Creativity: {is_creativity}")
        
        # Loop over X-axis metrics (Unexpectedness, Novelty)
        for x_name in X_METRICS:
            x_data = load_x_axis_data(excel_path, x_name, is_creativity)
            if x_data is None:
                continue
            
            # Genera grafici per ogni metrica Y
            for sheet_name, y_name in Y_METRICS.items():
                try:
                    y_data = load_metric_data(excel_path, sheet_name, is_creativity)
                    if y_data is None:
                        continue
                    
                    create_plot(y_name, y_data, x_name, x_data, dataset, method, OUTPUT_DIR)
                    total_plots += 1
                    
                except Exception as e:
                    print(f"❌ Errore per {y_name} vs {x_name}: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"🎉 Completato! Generati {total_plots} grafici in {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
