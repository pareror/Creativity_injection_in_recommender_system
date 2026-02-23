# Creativity_injection_in_recommender_system
Questa tesi si propone di superare limitazioni quali il Filter Bubble integrando nel processo di raccomandazione i principi della Creatività Computazionale, facendo riferimento al framework teorico di Margaret Boden. L'obiettivo è ingegnerizzare la
scoperta (Serendipity), bilanciando la rilevanza dei suggerimenti con metriche di Novelty e Unexpectedness, per offrire raccomandazioni che siano non solo utili, ma anche originali e sorprendenti.

# 🎯 Obiettivi
Gli obiettivi specici del lavoro possono essere così sintetizzati:
* **Formalizzazione della Creatività**: Tradurre i criteri qualitativi di Boden
(Novità, Sorpresa e Valore) in formule matematiche computabili.
* **Benchmarking della Creatività di Base**: Valutare le performance dei
modelli di raccomandazione standard (senza re-ranking) per stabilire una
linea base di quanto essi siano "naturalmente" creativi.
* **Sviluppo di Strategie di Re-ranking**: Proporre e confrontare due metodologie distinte per alterare l'ordinamento classico delle raccomandazioni:
    - Un approccio basato su una combinazione lineare di metriche (Creativity Score), che mira a sintetizzare le componenti della creatività in un unico punteggio scalare.
    - Un approccio (Re-ranking Bipolare), che agisce nello spazio latente per identificare item che soddisfano contemporaneamente criteri di vicinanza (rilevanza) e distanza (novità) rispetto al profilo utente.
* **Analisi del Trade-off**: Indagare sperimentalmente la relazione tra l'aumento delle metriche di creatività e la stabilità delle metriche di accuratezza
(NDCG, Recall, ...), al fine di identificare un punto di equilibrio ideale che massimizza la serendipità evitando la generazione di rumore.

# Scopo della repo
La seguente repo contiene tutti i codici utilizzati per:
* Fare tagli ai dataset
* Addestrare i modelli
* Valutare le metriche sui modelli addestrati

I modelli, i dataset e il loro addestramento e valutazione, fanno riferimento a [@Recbole](https://recbole.io/)

# 💻⚙️ Informazioni su hardware
* Processore (CPU): AMD FX-8320 Eight-Core Processor. Si tratta di una CPU dotata di 8 core.
* Memoria (RAM): 32 GB di memoria di sistema, supportati da 8 GB di memoria di swap.
* Acceleratori Grafici (GPU): Il sistema è dotato di una configurazione multi-GPU composta da due schede video NVIDIA GeForce GTX TITAN X.
  * Ciascuna GPU dispone di 12 GB di VRAM dedicata.
  * Versione Driver NVIDIA: 570.211.01.

Questa configurazione ha permesso di ridurre significativamente i tempi di addestramento, sfruttando l'accelerazione hardware CUDA

#🛠️ Requisiti
L'ambiente sperimentale è stato progettato per garantire la riproducibilità dei
risultati e l'efficienza computazionale. L'intera sperimentazione è stata eseguita su
un server, equipaggiato con il sistema operativo **Ubuntu 24.04.3 LTS (Noble
Numbat)** (Kernel Linux 6.8.0-90-generic)

Pre Requisiti:
* Python: 3.10.19+
* CUDA-compatible GPU (recommended)

## Usage
```
  # Clone the repository
  git clone https://github.com/pareror/Creativity_injection_in_recommender_system.git
  cd Creativity_injection_in_recommender_system

  # Create virtual environment
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  
  # Install dependencies
  pip install -r requirements.txt
```

# 📂 Struttura del progetto
```text
root/
│
├── Docs
│   └── Contiiene la documentazione del progetto (Tesi)
│
├── Excel & Grafici
│   └── Contiiene tutti i file csv delle valutazioni svolti. Inoltre contiene i grafici trand/trade-off/radio
│
├── requirements.txt
│   └── contiene le versioni delle librerie utilizzate
│
├── analyze_amazon_thresholds.py
│   └── fornisce le statistiche dei vari possibili tagli al dataset Amazon
│
├── create_amazon_cut.py
│   └── crea il dataset Amazon Books tagliato alla soglia indicata
│
├── train/   # FILE DI ADDESTRAMENTO
│   │
│   ├── train_and_save_recs.py
│   │   └── addestra e salva i modelli standard
│   │
│   └── train_and_save_recs_KG.py
│       └── addestra e salva i modelli (solo) KG
│           (in caso di problemi con train_and_save_recs.py)
│
└── eval/   # VALUTAZIONE
    │
    ├── eval_checkpoint_trainer_reranking_recbole.py
    │   └── valuta quasi tutti i modelli per il reranking bipolare (tranne enmf e lightgcn)
    ├── eval_reranking_enmf.py
    ├── eval_reranking_lightgcn.py
    │
    ├── eval_creativity_score_reranking.py
    │   └──v aluta quasi tutti i modelli per il metodo creativity score
    ├── eval_creativity_enmf.py
    └── eval_creativity_lightgcn.py
```
# 📊🤖 Informazioni su modelli e dataset
Gli esperimenti sono stati effettuati su i seguenti dataset e modelli di raccomandazione:

Dataset: amazon_books_500core e movielens_1m
## 📦 Dataset

| Dataset | Utenti | Item | Interazioni | Tipo |
|---|---|---|---|---|
| **MovieLens-1M** | 6.040 | 3.952 | ~1M | Rating espliciti (1–5) |
| **Amazon-Books (ridotto)** | 1.760 | 78.142 | ~900K | Rating espliciti, filtrati ≥500 interazioni/utente |

Entrambi i dataset sono arricchiti con **Knowledge Graph** (entità + relazioni semantiche) forniti tramite il formato Atomic Files di RecBole (`.inter`, `.item`, `.kg`, `.link`).

---
Modelli: Pop, ItemKNN, DMF, BPR, MultiVAE, LightGCN, ENMF, CFKG, CKE, KGCN, KGNNLS, MKR

# ⚙️ Metodologie

Entrambe le strategie seguono un'architettura **Two-Stage Retrieval & Re-ranking**: un modello standard genera i candidati, e un algoritmo di re-ranking li riordina per iniettare creatività.

## 1. Re-ranking Bipolare

Approccio innovativo che sostituisce la ricerca monopolare della rilevanza con un obiettivo **bipolare**:

- **Polo dell'Utilità (I_relevant):** il candidato più simile al profilo utente (centro della zona di comfort).
- **Polo della Scoperta (I_surprise):** il candidato che massimizza la distanza semantica pur restando sopra una soglia minima di rilevanza α.

Ogni candidato viene valutato in base a quanto funge da *ponte* tra i due poli:

$$Score_{seren}(i) = Sim(\vec{V}_i, \vec{V}_R) \times Sim(\vec{V}_i, \vec{V}_S)$$

E successivamente lo score assegnato permette di riordinare la lista di K raccomandazioni

## 2. Creativity Score

Punteggio composito che mappa direttamente le tre dimensioni di Boden tramite combinazione lineare pesata:

$$Score_{creative}(i) = w_{rel} \cdot \hat{S}_{rel}(i) + w_{nov} \cdot \hat{S}_{nov}(i) + w_{unexp} \cdot \hat{S}_{unexp}(i)$$

Dove:
- **Relevance** = score del modello (normalizzato)
- **Novelty** = popolarità inversa logaritmica: `1 / ln(1 + pop(i))`
- **Unexpectedness** = distanza coseno dal profilo utente: `1 − Sim(U, V_i)`

Tutte le componenti sono **normalizzate Min-Max** localmente sulla lista dei candidati. La formulazione com addizione permette un comportamento **compensativo** — un item eccellente in una dimensione può comunque posizionarsi bene anche se debole in un'altra.

### Configurazioni di Pesi Testate

| Configurazione | w_rel | w_nov | w_unexp |
|---|---|---|---|
| Bilanciata | 0.33 | 0.33 | 0.33 |
| Alta Inaspettatezza | 0.25 | 0.25 | 0.50 |
| Alta Novità | 0.25 | 0.50 | 0.25 |
| Alta Rilevanza | 0.50 | 0.25 | 0.25 |

---
# 📊 Metriche di Valutazione

| Metrica | Scopo |
|---|---|
| **Precision@K** | Frazione di item raccomandati effettivamente rilevanti |
| **Recall@K** | Frazione di item rilevanti effettivamente raccomandati |
| **NDCG@K** | Qualità del ranking con penalizzazione posizionale |
| **AveragePopularity@K** | Popolarità media di un item in una lista di raccomandazione |
| **Novelty** | Popolarità inversa logaritmica — promuove item long-tail |
| **Unexpectedness** | Distanza coseno media dal profilo utente |
| **Gini Index** | Equità della copertura del catalogo (più basso = più equo) |
| **Shannon Entropy** | Contenuto informativo della distribuzione delle raccomandazioni |
| **Serendipity (Ge)** | Insiemistica: item rilevanti *non* presenti nella lista "Most Popular" |
| **Serendipity (Yan)** | Continua: `rilevanza × inaspettatezza` |
| **Creativity Score** | Composita: `0.33·NDCG + 0.33·Novelty + 0.33·Unexpectedness` |

---

# 📈 Domande di Ricerca e Risultati Principali

## RQ1 — Quale modello raggiunge il miglior livello di creatività?

I modelli **Knowledge-Aware** (KGCN, KGNN-LS, MKR) dimostrano la creatività intrinseca più elevata, grazie alla capacità di sfruttare le connessioni semantiche del grafo di conoscenza per suggerire item meno ovvi. Tra i modelli collaborativi, **LightGCN** si distingue nettamente, affiancato da **ENMF** che ottiene punteggi competitivi su entrambi i dataset.

## RQ2 — Qual è l'impatto delle strategie di re-ranking sulle metriche beyond-accuracy?

Tutte le strategie producono un **miglioramento sistematico** nelle dimensioni beyond-accuracy, a fronte di un calo moderato di accuratezza:

- **Novità**: incremento significativo su tutti i modelli e dataset
- **Gini Index**: riduzione marcata → copertura più equa del catalogo
- **Inaspettatezza**: aumento modesto ma costante
- **Recall**: diminuisce proporzionalmente all'aggressività del re-ranking

Il dataset MovieLens beneficia in modo più consistente delle strategie di re-ranking rispetto ad AmazonBook, probabilmente per la maggiore densità di interazioni.

## RQ3 — Quale strategia offre il miglior trade-off?

Il **Creativity Score Bilanciato (λ=0.33)** ottiene il **miglior trade-off complessivo**:
- Rapporto di Efficienza per la Novità pari a **2.32** su AmazonBook (per ogni unità di Recall persa, si guadagnano 2.32 unità di Novità).
- Minimizza il calo di Serendipità meglio di qualsiasi altra strategia.
- Produce la riduzione più forte del Gini Index (distribuzione più equa del catalogo).

Il **Creativity Score Weighted (0.25-0.25-0.50)** è la scelta ottimale per **massimizzare l'Inaspettatezza** in modo specifico (rapporto di efficienza 1:1 su AmazonBook).

Il **Re-ranking Bipolare**, seppur concettualmente solido e più spiegabile, si è rivelato **eccessivamente conservativo** nel convertire la perdita di accuratezza in valore aggiunto beyond-accuracy.

> **Paradosso della Serendipità:** Tutte le strategie di re-ranking causano un lieve *calo* della Serendipità complessiva, poiché il guadagno in sorpresa non compensa completamente la perdita di rilevanza all'interno della formula moltiplicativa.

---

# 🔮 Sviluppi Futuri

L'evoluzione naturale di questo lavoro prevede l'integrazione di **Agenti AI Generativi** (LLM) direttamente nel loop di re-ranking. Anziché affidarsi a formule matematiche fisse, un LLM potrebbe ricevere in input lo storico completo dell'utente, la lista di candidati e metadati arricchiti, per effettuare un **re-ranking semantico** — selezionando item sulla base di ragionamento contestuale profondo anziché distanze numeriche. Questo consentirebbe il passaggio da una creatività *sintattica* (basata su formule) a una creatività **semantica** (basata sul significato).

---

# 👨‍🎓 Autore
> **Alessio Pagano**

Tesi:
> La Creatività nei Recommender Systems: Strategie di Re-ranking per la ricerca di un Trade-Off traAccuratezza e Creatività


Corso di Laurea in Informatica e Tecnologie per la produzione del software
Università degli Studi di Bari “Aldo Moro”

📫 Contatti: alexio3da@gmail.com

🔗 GitHub: https://github.com/pareror

> **Relatore:** Prof. Cataldo Musto | **Correlatrice:** Prof.ssa Allegra De Filippo  
> **Anno accademico** 2024 – 2025

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
