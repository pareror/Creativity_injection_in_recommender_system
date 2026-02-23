import os
import sys
import gc
import traceback
from collections import Counter, defaultdict

# --- FIX CRITICO SCIPY ---
try:
    import scipy.sparse as _sp
    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, data_dict):
            for k, v in data_dict.items():
                self[k] = v
            return self
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.quick_start import load_data_and_model  # ⭐ USA STESSA FUNZIONE DELLO SCRIPT GENERICO
from recbole.trainer import Trainer
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

# ==========================================
# CONFIGURAZIONE
# ==========================================

CHECKPOINTS = [
    #'./saved/ENMF-Dec-22-2025_16-15-22.pth',  # MovieLens (full, RO)
    #'./saved/ENMF-Dec-22-2025_18-49-08.pth',  # Amazon (full, RO) - sostituito uni100
    "./saved/ENMF-Jan-22-2026_19-37-08.pth", #MovieLens
    "./saved/ENMF-Jan-22-2026_21-11-40.pth", #Amazon
]

EVAL_BATCH_SIZE = 32
RESULTS_FILE = 'benchmark_reranking_enmf.csv'
LOG_FILE = 'eval_reranking_enmf.log'
ALPHA_FIXED = 0.5
TOPKS = [5, 10]

if not torch.cuda.is_available():
    print("⛔ GPU non trovata.")
    sys.exit(1)

DEVICE = torch.device("cuda")

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# ==========================================
# CARICAMENTO MODELLO - USA load_data_and_model COME SCRIPT GENERICO
# ==========================================

def custom_load_model(model_file):
    """
    Usa load_data_and_model di RecBole per garantire stessa metodologia
    di valutazione dello script generico.
    
    ⭐ FIX CRITICO: Prima usava Config() custom che forzava 'full' ranking,
    causando metriche non comparabili con altri modelli addestrati con 'uni100'.
    """
    print(f"   📂 Loading: {os.path.basename(model_file)}")
    
    # Usa la stessa funzione dello script generico
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)
    
    # Override solo parametri di runtime, NON di valutazione
    config['use_gpu'] = True
    config['device'] = DEVICE
    config['gpu_id'] = '0'
    config['eval_batch_size'] = EVAL_BATCH_SIZE
    config['topk'] = TOPKS
    config['metrics'] = ['NDCG', 'Recall', 'Precision', 'AveragePopularity', 'GiniIndex', 'ShannonEntropy']
    
    model = model.to(DEVICE)
    model.eval()
    
    # Restituisce 6 valori come load_data_and_model
    return config, model, dataset, train_data, valid_data, test_data

# ==========================================
# UTILITY
# ==========================================

def build_pop_counter(inter_feat, iid_field):
    """Allineato con script generico - prende inter_feat e iid_field."""
    pop = Counter()
    for iid in inter_feat[iid_field].numpy():
        pop[int(iid)] += 1
    return pop

def build_ground_truth(inter_feat, uid_field, iid_field, label_field=None):
    """
    Costruisce il ground truth dalle interazioni di test.
    
    Se label_field è presente (da threshold), filtra solo label=1.
    Altrimenti, usa tutte le interazioni.
    """
    truth = defaultdict(set)
    uids = inter_feat[uid_field].numpy()
    iids = inter_feat[iid_field].numpy()
    
    if label_field and label_field in inter_feat:
        # Filtra per label=1 (solo rating >= threshold)
        labels = inter_feat[label_field].numpy()
        for u, i, label in zip(uids, iids, labels):
            if label == 1:
                truth[int(u)].add(int(i))
    else:
        # Usa tutte le interazioni (threshold=None)
        for u, i in zip(uids, iids):
            truth[int(u)].add(int(i))
    
    return truth

def build_user_history_cpu(dataset):
    inter_feat = dataset.inter_feat
    users = inter_feat[dataset.uid_field].numpy()
    items = inter_feat[dataset.iid_field].numpy()
    history = defaultdict(list)
    for u, i in zip(users, items):
        history[u].append(i)
    out = defaultdict(lambda: torch.LongTensor([]))
    for u, arr in history.items():
        out[u] = torch.LongTensor(arr)
    return out

def get_vectors_for_yan(model):
    if hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding'):
        return model.user_embedding.weight.detach(), model.item_embedding.weight.detach()
    return None, None

# ==========================================
# GENERAZIONE RACCOMANDAZIONI (FIX ENMF)
# ==========================================

def generate_recs_for_reranking(model, test_data, topk, user_history):
    """Genera top-k raccomandazioni con fix per ENMF reshape E filtraggio history."""
    user_recs = {}
    uid_field = model.USER_ID
    
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating Lists", leave=False):
            interaction = batch[0].to(DEVICE)
            scores = model.full_sort_predict(interaction)
            
            # --- FIX CRITICO: RESHAPE SICURO PER ENMF ---
            batch_users = interaction[uid_field]
            current_batch_size = batch_users.size(0)
            
            if scores.dim() == 1:
                scores = scores.view(current_batch_size, -1)
            # -----------------------------------------------
            
            # ⭐⭐⭐ FILTRAGGIO HISTORY (ALLINEAMENTO CON SCRIPT GENERICO) ⭐⭐⭐
            batch_users_list = batch_users.tolist()
            n_items = scores.size(1)
            for i, u_id in enumerate(batch_users_list):
                seen = user_history[int(u_id)]
                if len(seen) == 0:
                    continue
                valid_mask = seen < n_items
                valid_seen = seen[valid_mask]
                if len(valid_seen) > 0:
                    scores[i, valid_seen.to(DEVICE)] = -float("inf")
            # ⭐⭐⭐ FINE FILTRAGGIO ⭐⭐⭐
            
            _, indices = torch.topk(scores, topk, dim=1)
            
            batch_users_cpu = batch_users.cpu().numpy()
            batch_recs_cpu = indices.cpu().numpy()
            
            for i, uid in enumerate(batch_users_cpu):
                user_recs[int(uid)] = batch_recs_cpu[i].tolist()
                
    return user_recs

# ==========================================
# RERANKING BIPOLARE
# ==========================================

def rerank_bipolar(user_recs, user_history, i_embs, num_candidates=100, topk=10, alpha_threshold=None):
    if i_embs is None:
        print("⚠️ Embedding item non disponibili, skip reranking")
        return user_recs
    
    device = i_embs.device
    i_embs_norm = F.normalize(i_embs, p=2, dim=1)
    user_recs_reranked = {}
    
    for uid, candidates in tqdm(user_recs.items(), desc="Reranking bipolare", leave=False):
        C = candidates[:num_candidates]
        
        if len(C) < 2:
            user_recs_reranked[uid] = C[:topk]
            continue
        
        history = user_history.get(int(uid), torch.LongTensor([]))
        if len(history) == 0:
            user_recs_reranked[uid] = C[:topk]
            continue
        
        valid_history = history[history < i_embs.shape[0]]
        if len(valid_history) == 0:
            user_recs_reranked[uid] = C[:topk]
            continue
        
        valid_history = valid_history.to(device).long()
        user_profile = i_embs_norm[valid_history].mean(dim=0)
        
        C_tensor = torch.LongTensor(C).to(device)
        C_embs = i_embs_norm[C_tensor]
        
        sims = F.cosine_similarity(user_profile.unsqueeze(0), C_embs, dim=1)
        sims = (sims + 1.0) / 2.0
        sims = torch.clamp(sims, 0.0, 1.0)
        
        idx_relevant = torch.argmax(sims).item()
        i_relevant = C[idx_relevant]
        
        if alpha_threshold is None:
            alpha = sims.mean().item()
        else:
            alpha = alpha_threshold
        
        valid_mask = sims >= alpha
        if not valid_mask.any():
            alpha = sims.min().item()
            valid_mask = sims >= alpha
        if not valid_mask.any():
            valid_mask[torch.argmin(sims)] = True
        
        valid_sims = sims.clone()
        valid_sims[~valid_mask] = float("inf")
        idx_surprise = torch.argmin(valid_sims).item()
        i_surprise = C[idx_surprise]
        
        if i_relevant == i_surprise:
            rerank_scores = sims ** 2
        else:
            V_relevant = i_embs_norm[i_relevant]
            V_surprise = i_embs_norm[i_surprise]
            
            sim_with_relevant = F.cosine_similarity(C_embs, V_relevant.unsqueeze(0), dim=1)
            sim_with_relevant = (sim_with_relevant + 1.0) / 2.0
            sim_with_relevant = torch.clamp(sim_with_relevant, 0.0, 1.0)
            
            sim_with_surprise = F.cosine_similarity(C_embs, V_surprise.unsqueeze(0), dim=1)
            sim_with_surprise = (sim_with_surprise + 1.0) / 2.0
            sim_with_surprise = torch.clamp(sim_with_surprise, 0.0, 1.0)
            
            rerank_scores = sim_with_relevant * sim_with_surprise
        
        _, rerank_indices = torch.sort(rerank_scores, descending=True)
        C_reranked = [C[i] for i in rerank_indices[:topk].cpu().numpy()]
        
        user_recs_reranked[uid] = C_reranked
    
    return user_recs_reranked

# ==========================================
# DATASTRUCT RECBOLE
# ==========================================

def build_recbole_datastruct_from_reranked(user_recs, ground_truth, train_dataset, topks):
    if not user_recs:
        return None

    users = [uid for uid in user_recs.keys() if uid in ground_truth and len(ground_truth[uid]) > 0]
    if not users:
        return None

    max_k = max(topks)
    item_matrix = torch.zeros((len(users), max_k), dtype=torch.long)
    pos_matrix = torch.zeros((len(users), max_k), dtype=torch.int)
    pos_len_list = torch.zeros((len(users), 1), dtype=torch.int)

    for row_idx, uid in enumerate(users):
        recs = user_recs[uid][:max_k]
        if len(recs) < max_k:
            recs = recs + ([recs[-1]] * (max_k - len(recs))) if recs else [0] * max_k
        item_matrix[row_idx] = torch.tensor(recs, dtype=torch.long)
        gt = ground_truth[uid]
        pos_len_list[row_idx, 0] = len(gt)
        hit_row = [1 if iid in gt else 0 for iid in recs[:max_k]]
        pos_matrix[row_idx] = torch.tensor(hit_row, dtype=torch.int)

    data_struct = DataStruct()
    data_struct.set("rec.items", item_matrix)
    data_struct.set("rec.topk", torch.cat([pos_matrix, pos_len_list], dim=1))
    data_struct.set("data.num_items", train_dataset.item_num)
    data_struct.set("data.num_users", train_dataset.user_num)
    data_struct.set("data.count_items", train_dataset.item_counter)
    data_struct.set("data.count_users", train_dataset.user_counter)
    return data_struct

# ==========================================
# SERENDIPITÀ
# ==========================================

def serendipity_ge_binary(user_recs, ground_truth, pop_counter, topk):
    if not user_recs:
        return 0.0
    pm_set = set([item for item, _ in pop_counter.most_common(topk)])
    total = 0.0
    users = 0
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        l_u = recs[:topk]
        t_u = ground_truth[uid]
        hits = [i for i in l_u if i in t_u]
        ser_items = [i for i in hits if i not in pm_set]
        total += len(ser_items) / float(topk)
        users += 1
    return total / users if users else 0.0

def calc_serendipity_and_unexpectedness_yan_gpu(user_recs, ground_truth, u_embs, i_embs, topk):
    if not user_recs or u_embs is None or i_embs is None:
        return 0.0, 0.0
    total_ser = 0.0
    total_unexp = 0.0
    users = 0
    u_norm = F.normalize(u_embs, p=2, dim=1).cpu()
    i_norm = F.normalize(i_embs, p=2, dim=1).cpu()
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        if uid >= len(u_norm):
            continue
        t_u = ground_truth[uid]
        l_u = recs[:topk]
        score_ser_u = 0.0
        score_unexp_u = 0.0
        user_vec = u_norm[uid]
        for iid in l_u:
            if iid >= len(i_norm):
                continue
            item_vec = i_norm[iid]
            sim = torch.dot(user_vec, item_vec).item()
            sim = (sim + 1.0) / 2.0
            sim = max(0.0, min(1.0, sim))
            unexp = 1.0 - sim
            score_unexp_u += unexp
            if iid in t_u:
                score_ser_u += unexp
        total_ser += score_ser_u / float(topk)
        total_unexp += score_unexp_u / float(topk)
        users += 1
    return (total_ser / users if users else 0.0), (total_unexp / users if users else 0.0)

# ==========================================
# VALUTAZIONE
# ==========================================

def evaluate_checkpoint(model_path):
    print(f"\n⚙️  Processing: {os.path.basename(model_path)}")
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # ⭐ Ora restituisce 6 valori come load_data_and_model
        config, model, dataset, train_data, valid_data, test_data = custom_load_model(model_file=model_path)
        
        print("   📊 Running RecBole Native Evaluator (baseline)...")
        trainer = Trainer(config, model)
        trainer.eval_collector.data_collect(train_data)
        base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        # ⭐ Usa stessa firma dello script generico
        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
        pop_counter = build_pop_counter(train_data.dataset.inter_feat, iid_field)
        ground_truth = build_ground_truth(test_data.dataset.inter_feat, uid_field, iid_field, label_field=label_field)
        user_history = build_user_history_cpu(train_data.dataset)
        
        print("   🔮 Generating 100 candidates for reranking (con filtraggio history)...")
        user_recs = generate_recs_for_reranking(model, test_data, 100, user_history)
        
        # Verifica che il filtraggio sia avvenuto correttamente
        seen_in_candidates_count = 0
        total_candidates = 0
        for uid, candidates in user_recs.items():
            history_set = set(user_history[int(uid)].numpy())
            for item in candidates[:10]:  # Check primi 10 per velocità
                total_candidates += 1
                if item in history_set:
                    seen_in_candidates_count += 1
        if total_candidates > 0:
            pct = 100 * seen_in_candidates_count / total_candidates
            print(f"   ✓ Items già visti nei candidati: {pct:.1f}% (dovrebbe essere ~0%)")
        
        u_emb, i_emb = get_vectors_for_yan(model)
        
        print(f"   🎯 Applying bipolar reranking (alpha={ALPHA_FIXED})...")
        user_recs_reranked = rerank_bipolar(
            user_recs,
            user_history,
            i_emb,
            num_candidates=100,
            topk=max(TOPKS),
            alpha_threshold=ALPHA_FIXED
        )
        
        result = {
            'Checkpoint': os.path.basename(model_path),
            'Dataset': config['dataset'],
            'Model': config['model'],
        }
        
        # Metriche RecBole originali
        for key, val in base_metrics.items():
            result[key] = val
        
        # Metriche RecBole sul reranking
        rerank_data_struct = build_recbole_datastruct_from_reranked(
            user_recs_reranked, ground_truth, train_data.dataset, TOPKS
        )
        if rerank_data_struct is not None:
            rerank_config = {
                'topk': TOPKS,
                'metrics': [m.lower() for m in config['metrics']],
                'metric_decimal_place': config['metric_decimal_place'],
            }
            evaluator = Evaluator(rerank_config)
            rerank_metrics = evaluator.evaluate(rerank_data_struct)
            for key, val in rerank_metrics.items():
                result[f'{key}_reranked'] = val
            
            # Aggiungi delta per tutte le metriche RecBole
            for key, val in base_metrics.items():
                if key in rerank_metrics:
                    result[f'Delta_{key}'] = rerank_metrics[key] - val
        
        # Serendipità custom
        for k in TOPKS:
            k_recs = {u: r[:k] for u, r in user_recs.items()}
            k_recs_reranked = {u: r[:k] for u, r in user_recs_reranked.items()}
            
            ser_ge_orig = serendipity_ge_binary(k_recs, ground_truth, pop_counter, k)
            ser_yan_orig, unexp_orig = calc_serendipity_and_unexpectedness_yan_gpu(k_recs, ground_truth, u_emb, i_emb, k)
            ser_ge_reranked = serendipity_ge_binary(k_recs_reranked, ground_truth, pop_counter, k)
            ser_yan_reranked, unexp_reranked = calc_serendipity_and_unexpectedness_yan_gpu(k_recs_reranked, ground_truth, u_emb, i_emb, k)
            
            result[f'Serendipity_Ge_Original@{k}'] = ser_ge_orig
            result[f'Serendipity_Yan_Original@{k}'] = ser_yan_orig
            result[f'Unexpectedness_Original@{k}'] = unexp_orig
            result[f'Serendipity_Ge_Reranked@{k}'] = ser_ge_reranked
            result[f'Serendipity_Yan_Reranked@{k}'] = ser_yan_reranked
            result[f'Unexpectedness_Reranked@{k}'] = unexp_reranked
            
            result[f'Delta_Serendipity_Ge@{k}'] = ser_ge_reranked - ser_ge_orig
            result[f'Delta_Serendipity_Yan@{k}'] = ser_yan_reranked - ser_yan_orig
            result[f'Delta_Unexpectedness@{k}'] = unexp_reranked - unexp_orig

        return result

    except Exception as e:
        print(f"❌ ERRORE: {e}")
        traceback.print_exc()
        return None

# ==========================================
# MAIN
# ==========================================

def main():
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        print("=" * 80)
        print("VALUTAZIONE RERANKING ENMF (alpha=0.1) CON METRICHE RECBOLE")
        print("=" * 80)
        
        results = []
        header_written = False
        for i, ck in enumerate(CHECKPOINTS, 1):
            if not os.path.exists(ck):
                print(f"File not found: {ck}")
                continue
            
            print(f"[{i}/{len(CHECKPOINTS)}]")
            res = evaluate_checkpoint(ck)
            if res:
                results.append(res)
                pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=not header_written, index=False, float_format='%.6f')
                header_written = True
            gc.collect()
            torch.cuda.empty_cache()
        
        if results:
            df = pd.DataFrame(results)
            cols = [c for c in ['Checkpoint', 'Dataset', 'Model'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in cols]
            cols += other_cols
            try:
                print(df[cols].to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                print(df[cols])
    finally:
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()

if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
