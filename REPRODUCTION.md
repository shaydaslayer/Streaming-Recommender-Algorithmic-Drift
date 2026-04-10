# Reproducing the code on a fresh Windows machine:

This reproduces the baseline RecVAE simulation and baseline evaluation, not the novel reranked version, to reproduce the novel reranked version there will be a note at the end. 

---

## Expected baseline results:

- HR@10 ≈ 0.3376  
- Recall@10 ≈ 0.0457  
- NDCG@10 ≈ 0.0463  
- Average TopK Entropy ≈ 0.7407  
- Average History Entropy ≈ 0.6683  

---

## Before you start:

Make sure that you have these on your machine:

- Python 3.9.x  
- Processed MovieLens-1M + RecVAE checkpoint zip/release asset  

---

## IMPORTANT:

**Do NOT download this repository as a ZIP file.**  
This project uses a Git submodule, and downloading as a ZIP will result in missing files.

You MUST clone the repository using:

```bash
git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

---

## Step 1 - Clone the repo with submodules:

Open command prompt and run: 

```bash
cd C:\Users\%USERNAME%
git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git
cd Streaming-Recommender-Algorithmic-Drift
```

This should create:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift
```

---

## Step 2 - Create and activate the Python environment:

From the repository root run:

```bash
py -3.9 -m venv venv39
venv39\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Your command prompt should now begin with:

```
(venv39)
```

If `py -3.9` is not recognized, install Python 3.9 from python.org and ensure it is added to PATH.

---

## Step 3 - Install needed dependencies:

From the repository root run:

```bash
pip install -r code\authors_implementation\AlgorithmicDrift\requirements.txt
pip uninstall -y networkx recbole
pip install networkx==2.8
pip install recbole==1.0.1
```

It’s incredibly necessary that this is done correctly as the code WILL NOT run otherwise, again the EXACT environment used for reproduction is:

- Python 3.9.x  
- networkx==2.8  
- recbole==1.0.1  

---

## Step 4 - Download and extract the processed dataset + checkpoint:

Download the release asset containing the processed MovieLens-1M data and RecVAE checkpoint, it’s the only release on the repo. After downloading, extract the zip so that this folder exists exactly here:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m
```

After extraction, confirm that these paths exist:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\Histories.tsv
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\Histories_for_comparison.tsv
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\RecVAE\model_checkpoint\
```

If those files/folders are not present, the baseline simulation will not run.

### DEBUG CHECK:

Run the following command to verify data is correctly placed:

```bash
dir data\processed\movielens-1m
```

You should see files like:

- Histories.tsv  
- Histories_for_comparison.tsv  
- RecVAE folder  

If not, the dataset was not placed correctly.

---

## Step 5 - Copy the modified baseline files into the parent submodule:

From the repository root, run:

```bash
copy /Y replication_scripts\modified_files\data_utils_modified.py code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\utils\data_utils.py

copy /Y replication_scripts\modified_files\graph_generation_modified.py code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\start\graph_generation.py
```

This is required because the baseline reproduction depends on the modified author files stored in `replication_scripts\modified_files\`.

---

## Step 6 - Build the true holdout file:

From the repository root, run:

```bash
python replication_scripts\build_true_holdout.py
```

This should make:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\true_holdout_last10.tsv
```

Expected console output will look similar to:

```
Saved: C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\true_holdout_last10.tsv
Users with exact last-10 holdout: ...
```

---

## Step 7 - Confirm the evaluation scripts point to the baseline folder:

Open `replication_scripts\evaluate_table1_simulation.py` and confirm that the simulation path points to the baseline folder:

```python
SIM_SEQ_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0",
    "sim_sequences.tsv"
)
```

Open `replication_scripts\evaluate_entropy.py` and confirm that the entropy path points to the baseline folder:

```python
ENTROPY_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0",
    "entropy_metrics.tsv"
)
```

For the baseline approach, these paths should be pointing to:

```
...eta_0.0
```

If you want to recreate the novel approach, these paths should be pointing to:

```
...eta_0.0_novelty_divrerank_0.7
```

or whatever lambda value is chosen (the 0.7 gets replaced depending on the lambda value). 

---

## Step 8 - Run the baseline simulation:

Change into the exact simulation folder:

```bash
cd code\authors_implementation\AlgorithmicDrift\src\2.0-RecModules\start
```

Then run the baseline command:

```bash
python handle_modules.py "%CD%\data\processed\\" movielens-1m RecVAE generation No_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0
```

This command generates the baseline RecVAE simulation output.

---

## Step 9 - Confirm the baseline simulation output was created:

After the simulation finishes, this file should exist:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\RecVAE\graphs\topk_10\1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0\sim_sequences.tsv
```

And for entropy evaluation, this file should also exist:

```
C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\RecVAE\graphs\topk_10\1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0\entropy_metrics.tsv
```

If either file is missing, the simulation did not complete correctly. Please also note that baseline simulation takes a good amount of time ~6-8 hours. It’s best to just let the simulation run until the necessary files are generated.

---

## Step 10 - Run the baseline Table 1 evaluation:

Return to the repository root:

```bash
cd C:\Users\<YOUR_USERNAME>\Streaming-Recommender-Algorithmic-Drift
```

Run the baseline Table 1 evaluation:

```bash
python replication_scripts\evaluate_table1_simulation.py
```

Expected approximate output:

```
Simulation metrics (Table 1 style):
HR@10     = 0.3376
Recall@10 = 0.0457
NDCG@10   = 0.0463
```

The parent paper values printed by the script are:

```
HR@10     = ~0.3393
Recall@10 = ~0.0462
NDCG@10   = ~0.0467
```

---

## Step 11 - Run the baseline entropy evaluation:

Run the baseline entropy evaluation:

```bash
python replication_scripts\evaluate_entropy.py
```

Expected approximate output:

```
Entropy summary:
Rows read            = 21285000
Average TopK Entropy = ~0.7407
Average Hist Entropy = ~0.6683
```

---

# TO REPRODUCE THE NOVEL APPROACH:

If you want to reproduce the novel approach, the process is very similar to the baseline, since the file `novel_graph_generation.py` is already included in the repository.

---

## Step 1 - Backup the baseline graph generation file:

Before making any changes, create a backup of the baseline version of `graph_generation.py`.

The file is located at:

```
code/authors_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph_generation.py
```

You can make a copy of it in the same folder and rename it to something like:

```
graph_generation_baseline_backup.py
```

---

## Step 2 - Replace the baseline graph generation file with the novel version:

Go to:

```
replication_scripts/novel_graph_generation.py
```

Copy the full contents of that file and paste them into:

```
code/authors_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph_generation.py
```

This should completely replace the baseline graph_generation.py code with the novelty-enabled version.

---

## Step 3 - Verify that the novel settings are enabled:

After replacing the file, open:

```
code/authors_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph_generation.py
```

and confirm that these two lines are present:

```python
enable_diversity_rerank = True
lambda_div = 0.7
```

If these lines are not present exactly like this, then the file is still using the baseline version and the novel reranking approach will not run.

---

## Step 4 - (Optional) Change the diversity weight:

The value:

```python
lambda_div = 0.7
```

controls how strongly diversity is weighted during reranking, higher values put more emphasis on diversity and lower values put more emphasis on recommendation relevance.

For the reported novel results in this project, the value should remain at 0.7, if you change this value, your results may differ from the reported novel results.

---

## Step 5 - Update the evaluation scripts to point to the novel output folder:

To evaluate the novel approach, the scripts must point to the novelty output directory instead of the baseline directory.

In `replication_scripts/evaluate_table1_simulation.py`, make sure the simulation path points to:

```python
SIM_SEQ_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0_novelty_divrerank_0.7",
    "sim_sequences.tsv"
)
```

In `replication_scripts/evaluate_entropy.py`, make sure the entropy path points to:

```python
ENTROPY_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0_novelty_divrerank_0.7",
    "entropy_metrics.tsv"
)
```

---

## Step 6 - Run the same simulation command:

After replacing graph_generation.py with the novel version, run the same simulation command used for the baseline:

```bash
python handle_modules.py "%CD%\data\processed\\" movielens-1m RecVAE generation No_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0
```

The difference is that graph_generation.py now contains the novelty-enabled reranking logic, so the generated results will be written to the novelty folder.

---

## Step 7 - Run the evaluation scripts:

After the simulation finishes, run:

```bash
python replication_scripts\evaluate_table1_simulation.py
python replication_scripts\evaluate_entropy.py
```

These scripts should now read from the novelty folder and produce the novel results.

Expected approximate novel output:

- HR@10 ≈ 0.3527  
- Recall@10 ≈ 0.0498  
- NDCG@10 ≈ 0.0509  
- Average TopK Entropy ≈ 0.8251  
- Average History Entropy ≈ 0.6746  

---

If you want to switch back to the baseline after running the novel version, restore the original baseline graph_generation.py file from the backup you created in Step 1.
