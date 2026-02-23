import os
import sys
import glob
import gc
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
from recbole.data.interaction import Interaction
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

# Patch for KnowledgeBasedDataLoader missing 'dataset' attribute
if not hasattr(KnowledgeBasedDataLoader, 'dataset'):
    setattr(KnowledgeBasedDataLoader, 'dataset', property(lambda self: self._dataset))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPKS = [5, 10]
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 128))
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", 64))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 10000))
RESULTS_FILE = "benchmark_checkpoints_trainer_reranking_recbole.csv"
LOG_FILE = "eval_checkpoints_trainer_reranking_recbole.log"
CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.join(SCRIPT_DIR, "saved"))
ALPHA_FIXED = 0.5  # scelto dall'analisi alpha

# Se vuoi limitare i checkpoint, popola questa lista
TARGET_CKPTS = {
    #-------------movielens---------
    #"Pop-Dec-22-2025_12-31-19.pth",
    #"ItemKNN-Dec-22-2025_12-33-14.pth",
    #"DMF-Dec-22-2025_12-48-05.pth",
    #"BPR-Dec-22-2025_14-02-50.pth",
    #"MultiVAE-Dec-22-2025_14-12-06.pth",
    #'CFKG-Dec-14-2025_12-44-41.pth',
    #'CKE-Dec-14-2025_12-51-14.pth',
    #'KGCN-Dec-14-2025_13-00-00.pth',
    #'KGNNLS-Dec-14-2025_13-16-17.pth',
    #'MKR-Dec-14-2025_13-44-17.pth',
    #'LightGCN-Dec-22-2025_14-23-36.pth',
    #-------------------nuovo rating threshold-------------
    "Pop-Jan-22-2026_17-26-13.pth",#
    "ItemKNN-Jan-22-2026_17-27-04.pth",#
    "DMF-Jan-22-2026_17-34-35.pth",#
    "BPR-Jan-22-2026_17-49-51.pth",#
    "MultiVAE-Jan-22-2026_17-53-25.pth",#
    "LightGCN-Jan-22-2026_18-38-19.pth",#
    #"ENMF-Jan-22-2026_19-37-08.pth", #da fare nel file a parte
    "CFKG-Jan-23-2026_01-34-40.pth",#
    "CKE-Jan-23-2026_02-00-38.pth",#
    "KGCN-Jan-23-2026_02-29-31.pth",#
    "KGNNLS-Jan-23-2026_03-06-29.pth",#
    "MKR-Jan-23-2026_03-57-06.pth",#

    #-------------amazon books cut500---------
    #"Pop-Dec-22-2025_13-21-09.pth",
    #"ItemKNN-Dec-22-2025_13-25-29.pth",
    #"DMF-Dec-22-2025_13-38-17.pth",
    #"BPR-Dec-22-2025_16-27-18.pth",
    #'MultiVAE-Dec-22-2025_16-34-28.pth',
    #'CFKG-Dec-23-2025_16-04-23.pth',
    #'CKE-Dec-23-2025_16-09-13.pth',
    #'KGCN-Dec-23-2025_16-13-07.pth',
    #'KGNNLS-Dec-23-2025_16-20-39.pth',
    #'MKR-Dec-23-2025_16-28-21.pth',
    #-------------------nuovo rating threshold-------------
    "Pop-Jan-22-2026_19-39-48.pth",#
    "ItemKNN-Jan-22-2026_19-43-16.pth",#
    "DMF-Jan-22-2026_19-50-13.pth",#
    "BPR-Jan-22-2026_19-59-00.pth",#
    "MultiVAE-Jan-22-2026_20-01-57.pth",#
    #"LightGCN-Jan-22-2026_20-07-03.pth", #da fare nel file a parte
    #"ENMF-Jan-22-2026_21-11-40.pth", #da fare nel file a parte
    "CFKG-Jan-23-2026_05-01-13.pth",#
    "CKE-Jan-23-2026_05-05-29.pth",#
    "KGCN-Jan-23-2026_05-09-12.pth",#
    "KGNNLS-Jan-23-2026_05-15-53.pth",#
    "MKR-Jan-23-2026_05-22-59.pth",#
}

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# SciPy shim
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

def build_pop_counter(inter_feat, iid_field):
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

def get_vectors(model, dataset):
    u_emb, i_emb = None, None
    device = next(model.parameters()).device

    def extract(obj):
        if isinstance(obj, torch.nn.Embedding):
            return obj.weight.detach()
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return None

    name = model.__class__.__name__
    if name == "DMF":
        try:
            batch_size = 256
            u_list, i_list = [], []
            for i in range(0, dataset.user_num, batch_size):
                end = min(i + batch_size, dataset.user_num)
                users = torch.arange(i, end, device=device)
                emb = model.get_user_embedding(users)
                emb = model.user_fc_layers(emb)
                u_list.append(emb.detach())
            for i in range(0, dataset.item_num, batch_size):
                end = min(i + batch_size, dataset.item_num)
                items = torch.arange(i, end, device=device)
                col_indices = model.history_user_id[items].flatten()
                row_indices = torch.arange(items.shape[0], device=device).repeat_interleave(model.history_user_id.shape[1], dim=0)
                matrix_01 = torch.zeros(items.shape[0], model.n_users, device=device)
                matrix_01.index_put_((row_indices, col_indices), model.history_user_value[items].flatten())
                emb = model.item_linear(matrix_01)
                emb = model.item_fc_layers(emb)
                i_list.append(emb.detach())
            u_emb = torch.cat(u_list, dim=0)
            i_emb = torch.cat(i_list, dim=0)
            return u_emb, i_emb
        except Exception:
            return None, None

    if hasattr(model, "user_embeddings_lookup"):
        u_emb = extract(model.user_embeddings_lookup)
    elif hasattr(model, "user_embedding"):
        u_emb = extract(model.user_embedding)
    elif hasattr(model, "user_embeddings"):
        u_emb = extract(model.user_embeddings)

    if hasattr(model, "item_embeddings_lookup"):
        i_emb = extract(model.item_embeddings_lookup)
    elif hasattr(model, "item_embedding"):
        i_emb = extract(model.item_embedding)
    elif hasattr(model, "item_embeddings"):
        i_emb = extract(model.item_embeddings)
    elif hasattr(model, "entity_embedding"):
        i_emb = extract(model.entity_embedding)

    return u_emb, i_emb

def _predict_full_set(model, batch_users, dataset, device):
    num_items = dataset.item_num
    batch_size = batch_users.size(0)
    all_items = torch.arange(num_items, device=device)
    scores_matrix = torch.zeros(batch_size, num_items, device=device)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    for i, user_id in enumerate(batch_users):
        item_chunk_size = 10000
        for start_idx in range(0, num_items, item_chunk_size):
            end_idx = min(start_idx + item_chunk_size, num_items)
            current_items = all_items[start_idx:end_idx]
            current_len = len(current_items)
            u_ids = user_id.repeat(current_len)
            interaction_dict = {uid_field: u_ids, iid_field: current_items}
            interaction = Interaction(interaction_dict).to(device)
            with torch.no_grad():
                pred = model.predict(interaction)
            scores_matrix[i, start_idx:end_idx] = pred
    return scores_matrix

def generate_recommendations_gpu(model, test_data, dataset, topk, device, user_history):
    model.eval()
    user_field = dataset.uid_field
    user_recs = {}
    model = model.to(device)
    model_name = model.__class__.__name__
    is_lightgcn = model_name == 'LightGCN'
    is_graph_model = model_name in ['LightGCN', 'NGCF', 'GCN', 'KGCN']
    cached_user_emb = None
    cached_item_emb = None

    if is_lightgcn:
        with torch.no_grad():
            if hasattr(model, 'restore_user_e'): model.restore_user_e = None
            if hasattr(model, 'restore_item_e'): model.restore_item_e = None
            try:
                cached_user_emb, cached_item_emb = model.computer()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()
            except AttributeError:
                print("⚠️ model.computer() non trovato, provo model.forward()...")
                cached_user_emb, cached_item_emb = model.forward()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()

    has_embed = hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding') and not is_graph_model
    if has_embed:
        item_embed = model.item_embedding.weight.to(device)
    else:
        item_embed = None

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Gen recs", leave=False):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            batch_users = batch[user_field]
            batch_size = batch_users.size(0)

            if is_lightgcn:
                batch_u_emb = cached_user_emb[batch_users]
                scores = torch.matmul(batch_u_emb, cached_item_emb.t())
                batch_users_list = batch_users.tolist()
                n_items = scores.size(1)
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                _, topk_idx = torch.topk(scores, topk, dim=1)
            elif has_embed and dataset.item_num > 50000:
                user_e = model.user_embedding(batch_users).to(device)
                n_items = item_embed.size(0)
                vals = torch.empty((batch_size, 0), device=device)
                idxs = torch.empty((batch_size, 0), device=device, dtype=torch.long)
                for start in range(0, n_items, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_items)
                    scores_chunk = torch.matmul(user_e, item_embed[start:end].t())
                    batch_users_list = batch_users.tolist()
                    for i, u_id in enumerate(batch_users_list):
                        seen = user_history[int(u_id)]
                        if len(seen) == 0:
                            continue
                        mask_in_chunk = (seen >= start) & (seen < end)
                        if mask_in_chunk.any():
                            local_indices = (seen[mask_in_chunk] - start).to(device)
                            scores_chunk[i, local_indices] = -float("inf")
                    v, i_idx = torch.topk(scores_chunk, k=min(topk, end - start), dim=1)
                    i_idx = i_idx + start
                    vals = torch.cat([vals, v], dim=1)
                    idxs = torch.cat([idxs, i_idx], dim=1)
                    vals, top_idx = torch.topk(vals, k=topk, dim=1)
                    idxs = torch.gather(idxs, 1, top_idx)
                    del scores_chunk, v, i_idx
                topk_idx = idxs
            else:
                try:
                    scores = model.full_sort_predict(batch)
                except NotImplementedError:
                    scores = _predict_full_set(model, batch_users, dataset, device)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                if scores.size(0) != batch_size:
                    scores = scores.view(batch_size, -1)
                n_items = scores.size(1)
                batch_users_list = batch_users.tolist()
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                _, topk_idx = torch.topk(scores, topk, dim=1)

            uids = batch[user_field].cpu().numpy()
            topk_idx = topk_idx.cpu().numpy()
            for i, uid in enumerate(uids):
                user_recs[int(uid)] = [int(x) for x in topk_idx[i]]
            if 'scores' in locals():
                del scores
            del batch, topk_idx

    if is_lightgcn:
        if hasattr(model, 'restore_user_e'):
            model.restore_user_e = None
        if hasattr(model, 'restore_item_e'):
            model.restore_item_e = None
        del cached_user_emb, cached_item_emb

    return user_recs

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

def build_recbole_datastruct_from_reranked(user_recs, ground_truth, train_dataset, topks):
    """Costruisce un DataStruct RecBole dai risultati rerankati.

    Usa `rec.topk` (hit matrix + pos_len) e `rec.items` per permettere a Evaluator di
    calcolare le stesse metriche di RecBole sul ranking rerankato.
    """
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
            # pad con l'ultimo elemento per mantenere la forma attesa (recbole richiede k elementi)
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

# -------------------- Valutazione --------------------

def evaluate_checkpoint(model_path):
    print(f"\nProcessing: {model_path}")
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    try:
        config, model, dataset, train_data, _, test_data = load_data_and_model(model_file=model_path)
        config['use_gpu'] = USE_GPU
        config['device'] = DEVICE
        config['gpu_id'] = 0 if USE_GPU else ''
        config['eval_batch_size'] = EVAL_BATCH_SIZE
        config['topk'] = TOPKS
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = EVAL_BATCH_SIZE
            test_data.config['device'] = DEVICE
        object.__setattr__(test_data, 'batch_size', EVAL_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', EVAL_BATCH_SIZE)
        model = model.to(DEVICE)
        trainer = Trainer(config, model)
        trainer.save_model = False
        print("   Metriche RecBole (originali)...")
        trainer.eval_collector.data_collect(train_data)
        base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
        pop_counter = build_pop_counter(train_data.dataset.inter_feat, iid_field)
        test_truth = build_ground_truth(test_data.dataset.inter_feat, uid_field, iid_field, label_field=label_field)
        u_embs, i_embs = get_vectors(model, train_data.dataset)
        user_history = build_user_history_cpu(train_data.dataset)

        object.__setattr__(test_data, 'batch_size', GEN_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', GEN_BATCH_SIZE)
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = GEN_BATCH_SIZE
            test_data.config['device'] = DEVICE

        print("   Generando raccomandazioni (Retrieval)...")
        NUM_CANDIDATES = 100
        user_recs = generate_recommendations_gpu(model, test_data, dataset, NUM_CANDIDATES, DEVICE, user_history)

        print(f"   Applicando reranking bipolare (alpha={ALPHA_FIXED})...")
        user_recs_reranked = rerank_bipolar(
            user_recs,
            user_history,
            i_embs,
            num_candidates=NUM_CANDIDATES,
            topk=max(TOPKS),
            alpha_threshold=ALPHA_FIXED
        )

        try:
            dataset_name = config['dataset']
        except KeyError:
            dataset_name = dataset.name if hasattr(dataset, 'name') else ''
        try:
            model_name = config['model']
        except KeyError:
            model_name = model.__class__.__name__
        result = {
            'Checkpoint': os.path.basename(model_path),
            'Dataset': dataset_name,
            'Model': model_name,
        }

        # Metriche RecBole originali (valori di base)
        for key, val in base_metrics.items():
            result[key] = val

        # Metriche RecBole calcolate sul ranking rerankato
        rerank_data_struct = build_recbole_datastruct_from_reranked(
            user_recs_reranked, test_truth, train_data.dataset, TOPKS
        )
        if rerank_data_struct is not None:
            # Config minimale per Evaluator: solo ciò che serve alle metriche top-k
            rerank_config = {
                'topk': TOPKS,
                'metrics': [m.lower() for m in config['metrics']],
                'metric_decimal_place': config['metric_decimal_place'] if 'metric_decimal_place' in config else 4,
            }
            evaluator = Evaluator(rerank_config)
            rerank_metrics = evaluator.evaluate(rerank_data_struct)
            for key, val in rerank_metrics.items():
                result[f'{key}_reranked'] = val
            
            # Aggiungi delta per tutte le metriche RecBole
            for key, val in base_metrics.items():
                if key in rerank_metrics:
                    result[f'Delta_{key}'] = rerank_metrics[key] - val

        # Metriche di serendipità/inaspettatezza (restano personalizzate)
        for k in TOPKS:
            k_recs = {u: r[:k] for u, r in user_recs.items()}
            k_recs_reranked = {u: r[:k] for u, r in user_recs_reranked.items()}

            ser_ge_orig = serendipity_ge_binary(k_recs, test_truth, pop_counter, k)
            ser_yan_orig, unexp_orig = calc_serendipity_and_unexpectedness_yan_gpu(k_recs, test_truth, u_embs, i_embs, k)
            ser_ge_reranked = serendipity_ge_binary(k_recs_reranked, test_truth, pop_counter, k)
            ser_yan_reranked, unexp_reranked = calc_serendipity_and_unexpectedness_yan_gpu(k_recs_reranked, test_truth, u_embs, i_embs, k)

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
        print(f"Errore su {model_path}: {e}")
        traceback.print_exc()
        return None

def filter_checkpoints(ckpts):
    if not TARGET_CKPTS:
        return ckpts
    filtered = [p for p in ckpts if os.path.basename(p) in TARGET_CKPTS]
    missing = sorted(list(TARGET_CKPTS - {os.path.basename(p) for p in ckpts}))
    if missing:
        print("Checkpoint richiesti mancanti:", ", ".join(missing))
    return filtered

def main():
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee
    sys.stderr = tee
    try:
        print("GPU:", USE_GPU, "Device:", DEVICE)
        print("=" * 80)
        print(f"VALUTAZIONE RERANKING (alpha={ALPHA_FIXED}) CON METRICHE RECBOLE PARALLELE")
        print("=" * 80)
        ckpts = glob.glob(os.path.join(CKPT_ROOT, "**", "*.pth"), recursive=True)
        ckpts = filter_checkpoints(ckpts)
        if not ckpts:
            print("Nessun checkpoint trovato.")
            return
        results = []
        header_written = False
        for i, ck in enumerate(ckpts, 1):
            print(f"[{i}/{len(ckpts)}]")
            res = evaluate_checkpoint(ck)
            if res:
                results.append(res)
                pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=not header_written, index=False, float_format='%.6f')
                header_written = True
            gc.collect()
            if USE_GPU:
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

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
