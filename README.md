# Streaming Recommender Systems & Algorithmic Drift

**Group 38 – Shailee Patel**

## Background

This project reproduces and extends the simulation framework from
"Algorithmic Drift: A Simulation Framework to Study the Effects of Recommender Systems on User Preferences."

The goal is to analyze whether increasing recommendation diversity can reduce
filter bubble formation and influence long-term user behavior.

## Research Question
Do personalized recommendation systems in streaming services reinforce existing viewing habits or promote exploration of new content?

## Parent Paper
Algorithmic Drift: A simulation framework to study the effects of recommender systems on user preferences  
Coppolillo et al., 2025  
https://www.sciencedirect.com/science/article/pii/S0306457325000676

## Data
Processed datasets used by the paper:  
https://github.com/SimoneMungari/AlgorithmicDrift/tree/main/data/processed

## Parent paper code (submodule)
This repo includes the authors' implementation as a git submodule.

To clone with submodules:

```bash
git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git
```

## Repo Structure
- `paper/` – parent paper details and references
- `data/` – dataset access information
- `code/` – implementation and modified simulation code

---

## Project Status
This project reproduces the original simulation pipeline from the parent paper and extends it with additional metrics.

Current progress includes:
- Successfully reproducing the original simulation pipeline
- Running the RecVAE recommender model using the MovieLens-1M dataset
- Generating user navigation graphs from the simulation
- Implementing diversity-aware recommendation analysis
- Logging diversity metrics during simulation

---

## Novel Contribution
This project extends the original Algorithmic Drift framework by introducing diversity-aware recommendation analysis.

The following modifications were implemented:
- Category entropy metrics to measure diversity within recommendation lists
- Diversity-aware reranking of Top-K recommendations
- Logging of diversity metrics during simulation
- Tools for analyzing how recommendation diversity influences long-term user behavior

These additions allow the framework to evaluate whether increasing recommendation diversity can reduce filter bubble effects AND slow algorithmic drift.

---

## Reproducibility & Environment Setup (Windows)

This project was run and debugged on **Windows 11** using **Python 3.9**

### Python Environment
A dedicated virtual environment was used:

- Python version: **3.9.10**
- Virtual environment: `venv39`

To activate the environment:

```bash
.\venv39\Scripts\activate.bat
```

All required dependencies are listed in:

`code/authors_implementation/AlgorithmicDrift/requirements.txt`

However, `networkx` was downgraded to `networkx==2.8` to fix a dependency issue, and `recbole==1.0.1` was installed separately.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Code Modifications (Author Submodule)

To successfully run the paper’s simulation and generate graphs, several fixes and adjustments were required in the authors’ implementation.

These changes are **fully tracked in Git history**.

### Files Modified

#### 1. `models/recvae.py`
- Ensured the `predict_for_graphs()` method exists and functions correctly
- Fixed tensor indexing to ensure compatibility with PyTorch on Windows
- Ensured item indices are explicitly cast to `long` tensors
- Verified that the method returns flattened prediction scores for graph generation

This method is required for the **rec-guided simulation** stage used to build transition graphs.

---

#### 2. `start/graph_generation.py`
- Added category entropy metrics to measure recommendation diversity
- Implemented diversity-aware reranking for Top-K recommendations
- Added metric logging to track diversity and filter bubble behavior over time
- Limited the number of simulation iterations (`B` and `d`) for practical runtime during experimentation
- Removed excessive debug logging that caused runaway console output
- Ensured graph outputs are always written as:
  - `*_edge.tsv`
  - `*_node.tsv`
- Verified normalization of transition matrices before graph export

This file is responsible for producing **network graphs representing user navigation behavior**.

---

#### 3. `start/main.py`
- Fixed incorrect assumptions about execution paths
- Confirmed the correct invocation of `handle_modules.py`
- Ensured the pipeline executes:
  1. Dataset preparation
  2. Model loading
  3. Graph generation

---

#### 4. `.gitignore`
Added to prevent committing:
- Virtual environments (`venv/`, `venv39/`)
- Cached files (`__pycache__/`)
- TensorBoard logs
- Generated data and graph outputs

---

### Why These Changes Were Necessary
The original codebase assumes a Linux-based environment and longer simulation runs.

These adjustments were required to:
- Run the code reliably on Windows
- Prevent infinite or excessively long simulations

---

## Running the Simulation

All commands should be run from the repository root.

### Activate environment

```bash
.\venv39\Scripts\activate.bat
```

### Run the main pipeline

```bash
python src/2.0-RecModules/start/main.py
```

If successful, the process will:
- Load the MovieLens-1M dataset
- Run a RecVAE-guided simulation
- Generate graph files for each user

---

## Replication Results (Table 1 Verification)

To verify the parent paper, I implemented a full evaluation pipeline using:

- Full user histories (not truncated)
- True holdout (last 10 interactions per user)
- RecVAE simulation outputs (sim_sequences.tsv)
- Proper item ID mapping from internal to original MovieLens IDs

### Evaluation Script

Located in:
replication_scripts/evaluate_table1_simulation.py

### Final Results

Metric        | This Work | Parent Paper
-------------|----------|-------------
HR@10        | 0.3376   | 0.3393
Recall@10    | 0.0457   | 0.0462
NDCG@10      | 0.0463   | 0.0467

### Conclusion

The reproduced results closely match the parent paper, confirming:

- Correct dataset usage
- Proper simulation pipeline execution
- Correct evaluation methodology

---

## Replication Scripts

All custom scripts used for replication are in:

replication_scripts/

Includes:
- evaluate_table1_simulation.py
- build_true_holdout.py
- extract_item_mapping.py
- build_ground_truth_from_inter.py
- check_histories_split.py
- novel_graph_generation.py

Modified author files are stored in:

replication_scripts/modified_files/

Includes:
- graph_generation_modified.py
- data_utils_modified.py

---

## Note on Large Files

The file:

sim_sequences.tsv

is around 300MB and exceeds GitHub's file size limit, so it's not included in this repository.

To regenerate it, run the simulation pipeline:

python handle_modules.py "C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\\" movielens-1m RecVAE generation No_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0

This will recreate the simulation outputs used for evaluation.

---
