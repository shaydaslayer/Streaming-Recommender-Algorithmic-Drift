# Streaming Recommender Systems & Algorithmic Drift

**Group 36 – Shailee Patel**

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

## Repo Structure
- `paper/` – parent paper details and references  
- `data/` – dataset access information
- `code/` – any code worked on

---

## Reproducibility & Environment Setup (Windows)

This project was run and debugged on **Windows 10** using **Python 3.9**

### Python Environment
A dedicated virtual environment was used:

- Python version: **3.9.10**
- Virtual environment: `venv39`

To activate the environment:
```bash
.\venv39\Scripts\activate.bat

All required dependencies are listed in:
code/authors_implementation/AlgorithmicDrift/requirements.txt

However, networkx version was changed to networkx==2.8 to fix dependency issue, and recbole==1.0.1 was installed separately.

Install
pip install -r requirements.txt

## Code Modifications (Author Submodule)

To successfully run the paper’s simulation and generate graphs, several fixes and adjustments were required in the authors’ implementation.

These changes are **fully tracked in Git history** and preserved via a forked submodule.

### Files Modified

#### 1. `models/recvae.py`
- Ensured the `predict_for_graphs()` method exists and functions correctly
- Fixed tensor indexing to ensure compatibility with PyTorch on Windows
- Ensured item indices are explicitly cast to `long` tensors
- Verified that the method returns flattened prediction scores for graph generation

This method is required for the **rec-guided simulation** stage used to build transition graphs.

---

#### 2. `start/graph_generation.py`
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

## Running the Simulation
All commands should be run from:

### Activate environment
```bash
.\venv39\Scripts\activate.bat

Run the main pipeline:
python src/2.0-RecModules/start/main.py

If successful, the process will:
-Load the MovieLens-1M dataset
-Run a RecVAE-guided simulation
-Generate graph files for each user