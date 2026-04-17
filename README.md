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

IMPORTANT: Do NOT download this repository as a ZIP file

This repo includes the authors' implementation as a git submodule.

To clone with submodules:

```bash
git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Repo Structure
- `paper/` – parent paper details and references
- `data/` – dataset access information (download via release)
- `code/` – parent implementation (submodule)
- `replication_scripts/` – evaluation and custom scripts

---

## Full Reproduction Guide

For full step-by-step instructions, see:

```
REPRODUCTION.md
```

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

## Results

| Metric | Parent Paper | Baseline | Novel Approach |
|--------|------------|----------|----------------|
| HR@10 | 0.3393 | 0.3376 | **0.3527** |
| Recall@10 | 0.0462 | 0.0457 | **0.0498** |
| NDCG@10 | 0.0467 | 0.0463 | **0.0509** |
| Avg TopK Entropy | – | 0.7407 | **0.8251** |
| Avg History Entropy | – | 0.6683 | **0.6746** |

### Summary

The novel approach improves both recommendation accuracy and diversity.  
This demonstrates that increasing diversity does not necessarily reduce performance and can lead to more balanced long-term user behavior.

---

## Reproducibility & Environment Setup (Windows)

This project was run and debugged on **Windows 11** using **Python 3.9**

### Python Environment

- Python version: **3.9.10**
- Virtual environment: `venv39`

Activate environment:

```bash
.\venv39\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r code/authors_implementation/AlgorithmicDrift/requirements.txt
pip uninstall -y networkx recbole
pip install networkx==2.8
pip install recbole==1.0.1
```

---

## Running the Simulation

The simulation must be run from:

```
code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\start
```

### Activate environment

```bash
.\venv39\Scripts\activate.bat
```

### Change to simulation directory

```bash
cd code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\start
```

### Run simulation

```bash
python handle_modules.py "C:\Users\%USERNAME%\Streaming-Recommender-Algorithmic-Drift\data\processed\\" movielens-1m RecVAE generation No_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0
```

This will:
- Load MovieLens-1M data  
- Run RecVAE simulation  
- Generate user interaction sequences and graphs  

---

## Replication Results (Table 1 Verification)

Return to the repository root before running evaluation:

```bash
cd C:\Users\%USERNAME%\Streaming-Recommender-Algorithmic-Drift
```

Evaluation script:

```
replication_scripts/evaluate_table1_simulation.py
```

Run:

```bash
python replication_scripts\evaluate_table1_simulation.py
```

Entropy evaluation:

```bash
python replication_scripts\evaluate_entropy.py
```

---

## Replication Scripts

Located in:

```
replication_scripts/
```

Includes:
- evaluate_table1_simulation.py
- evaluate_entropy.py
- build_true_holdout.py
- novel_graph_generation.py

---

## Note on Large Files

The file:

```
sim_sequences.tsv
```

is too large for GitHub and is not included.

To regenerate it, run:

```bash
cd code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\start
python handle_modules.py "C:\Users\%USERNAME%\Streaming-Recommender-Algorithmic-Drift\data\processed\\" movielens-1m RecVAE generation No_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0
```

---
