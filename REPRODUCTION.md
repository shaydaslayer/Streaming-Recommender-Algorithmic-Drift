\# Reproducing the Code on a Fresh Windows Machine



This guide reproduces the \*\*baseline RecVAE simulation and baseline evaluation\*\*, not the novel reranked version.  

To reproduce the \*\*novel reranked version\*\*, there is a separate note at the end.



\---



\## Expected Baseline Results



\- \*\*HR@10 ≈ 0.3376\*\*

\- \*\*Recall@10 ≈ 0.0457\*\*

\- \*\*NDCG@10 ≈ 0.0463\*\*

\- \*\*Average TopK Entropy ≈ 0.7407\*\*

\- \*\*Average History Entropy ≈ 0.6683\*\*



\---



\## Before You Start



Make sure that you have these on your machine:



\- \*\*Python 3.9.x\*\*

\- \*\*Processed MovieLens-1M + RecVAE checkpoint zip/release asset\*\*



\---



\## IMPORTANT



\*\*Do NOT download this repository as a ZIP file.\*\*  

This project uses a Git submodule, and downloading as a ZIP will result in missing files.



You \*\*must\*\* clone the repository using:



```bash

git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git

```



If you already cloned without submodules, run:



```bash

git submodule update --init --recursive

```



\---



\## Step 1 - Clone the Repo with Submodules



Open Command Prompt and run:



```bash

cd C:\\Users\\%USERNAME%

git clone --recurse-submodules https://github.com/shaydaslayer/Streaming-Recommender-Algorithmic-Drift.git

cd Streaming-Recommender-Algorithmic-Drift

```



This should create:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift

```



\---



\## Step 2 - Create and Activate the Python Environment



From the repository root run:



```bash

py -3.9 -m venv venv39

venv39\\Scripts\\activate

python -m pip install --upgrade pip setuptools wheel

```



Your command prompt should now begin with:



```text

(venv39)

```



If `py -3.9` is not recognized, install Python 3.9 from python.org and ensure it is added to PATH.



\---



\## Step 3 - Install Needed Dependencies



From the repository root run:



```bash

pip install -r code\\authors\_implementation\\AlgorithmicDrift\\requirements.txt

pip uninstall -y networkx recbole

pip install networkx==2.8

pip install recbole==1.0.1

```



It is incredibly necessary that this is done correctly because the code will not run otherwise.



The exact environment used for reproduction is:



\- \*\*Python 3.9.x\*\*

\- \*\*networkx==2.8\*\*

\- \*\*recbole==1.0.1\*\*



\---



\## Step 4 - Download and Extract the Processed Dataset + Checkpoint



Download the release asset containing the processed MovieLens-1M data and RecVAE checkpoint.  

It is the only release on the repo.



After downloading, extract the zip so that this folder exists exactly here:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m

```



After extraction, confirm that these paths exist:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\Histories.tsv

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\Histories\_for\_comparison.tsv

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\RecVAE\\model\_checkpoint\\

```



If those files/folders are not present, the baseline simulation will not run.



\### Debug Check



Run the following command to verify data is correctly placed:



```bash

dir data\\processed\\movielens-1m

```



You should see files like:



\- `Histories.tsv`

\- `Histories\_for\_comparison.tsv`

\- `RecVAE` folder



If not, the dataset was not placed correctly.



\---



\## Step 5 - Copy the Modified Baseline Files into the Parent Submodule



From the repository root, run:



```bash

copy /Y replication\_scripts\\modified\_files\\data\_utils\_modified.py code\\authors\_implementation\\AlgorithmicDrift\\src\\2.0-RecModules\\utils\\data\_utils.py

copy /Y replication\_scripts\\modified\_files\\graph\_generation\_modified.py code\\authors\_implementation\\AlgorithmicDrift\\src\\2.0-RecModules\\start\\graph\_generation.py

```



This is required because the baseline reproduction depends on the modified author files stored in `replication\_scripts\\modified\_files\\`.



\---



\## Step 6 - Build the True Holdout File



From the repository root, run:



```bash

python replication\_scripts\\build\_true\_holdout.py

```



This should create:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\true\_holdout\_last10.tsv

```



Expected console output will look similar to:



```text

Saved: C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\true\_holdout\_last10.tsv

Users with exact last-10 holdout: ...

```



\---



\## Step 7 - Confirm the Evaluation Scripts Point to the Baseline Folder



Open `replication\_scripts\\evaluate\_table1\_simulation.py` and confirm that the simulation path points to the baseline folder:



```python

SIM\_SEQ\_PATH = os.path.join(

&#x20;   BASE\_DIR,

&#x20;   "RecVAE",

&#x20;   "graphs",

&#x20;   "topk\_10",

&#x20;   "1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0",

&#x20;   "sim\_sequences.tsv"

)

```



Open `replication\_scripts\\evaluate\_entropy.py` and confirm that the entropy path points to the baseline folder:



```python

ENTROPY\_PATH = os.path.join(

&#x20;   BASE\_DIR,

&#x20;   "RecVAE",

&#x20;   "graphs",

&#x20;   "topk\_10",

&#x20;   "1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0",

&#x20;   "entropy\_metrics.tsv"

)

```



For the baseline approach, these paths should be pointing to:



```text

...eta\_0.0

```



If you want to recreate the novel approach, these paths should be pointing to:



```text

...eta\_0.0\_novelty\_divrerank\_0.7

```



or whatever lambda value is chosen. The `0.7` gets replaced depending on the lambda value.



\---



\## Step 8 - Run the Baseline Simulation



Change into the exact simulation folder:



```bash

cd code\\authors\_implementation\\AlgorithmicDrift\\src\\2.0-RecModules\\start

```



Then run the baseline command:



```bash

python handle\_modules.py "%CD%\\data\\processed\\\\" movielens-1m RecVAE generation No\_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0

```



This command generates the baseline RecVAE simulation output.



\---



\## Step 9 - Confirm the Baseline Simulation Output Was Created



After the simulation finishes, this file should exist:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\RecVAE\\graphs\\topk\_10\\1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0\\sim\_sequences.tsv

```



And for entropy evaluation, this file should also exist:



```text

C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift\\data\\processed\\movielens-1m\\RecVAE\\graphs\\topk\_10\\1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0\\entropy\_metrics.tsv

```



If either file is missing, the simulation did not complete correctly.



Please also note that baseline simulation takes a good amount of time, approximately \*\*6–8 hours\*\*.  

It is best to just let the simulation run until the necessary files are generated.



\---



\## Step 10 - Run the Baseline Table 1 Evaluation



Return to the repository root:



```bash

cd C:\\Users\\<YOUR\_USERNAME>\\Streaming-Recommender-Algorithmic-Drift

```



Run the baseline Table 1 evaluation:



```bash

python replication\_scripts\\evaluate\_table1\_simulation.py

```



Expected approximate output:



```text

Simulation metrics (Table 1 style):

HR@10     = 0.3376

Recall@10 = 0.0457

NDCG@10   = 0.0463

```



The parent paper values printed by the script are:



```text

HR@10     = \~0.3393

Recall@10 = \~0.0462

NDCG@10   = \~0.0467

```



\---



\## Step 11 - Run the Baseline Entropy Evaluation



Run the baseline entropy evaluation:



```bash

python replication\_scripts\\evaluate\_entropy.py

```



Expected approximate output:



```text

Entropy summary:

Rows read            = 21285000

Average TopK Entropy = \~0.7407

Average Hist Entropy = \~0.6683

```



\---



\# To Reproduce the Novel Approach



If you want to reproduce the novel approach, the process is very similar to the baseline, since the file `novel\_graph\_generation.py` is already included in the repository.



\---



\## Step 1 - Back Up the Baseline Graph Generation File



Before making any changes, create a backup of the baseline version of `graph\_generation.py`.



The file is located at:



```text

code/authors\_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph\_generation.py

```



You can make a copy of it in the same folder and rename it to something like:



```text

graph\_generation\_baseline\_backup.py

```



\---



\## Step 2 - Replace the Baseline Graph Generation File with the Novel Version



Go to:



```text

replication\_scripts/novel\_graph\_generation.py

```



Copy the full contents of that file and paste them into:



```text

code/authors\_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph\_generation.py

```



This should completely replace the baseline `graph\_generation.py` code with the novelty-enabled version.



\---



\## Step 3 - Verify That the Novel Settings Are Enabled



After replacing the file, open:



```text

code/authors\_implementation/AlgorithmicDrift/src/2.0-RecModules/start/graph\_generation.py

```



and confirm that these two lines are present:



```python

enable\_diversity\_rerank = True

lambda\_div = 0.7

```



If these lines are not present exactly like this, then the file is still using the baseline version and the novel reranking approach will not run.



\---



\## Step 4 - Optional: Change the Diversity Weight



The value:



```python

lambda\_div = 0.7

```



controls how strongly diversity is weighted during reranking.



\- Higher values put more emphasis on diversity

\- Lower values put more emphasis on recommendation relevance



For the reported novel results in this project, the value should remain at `0.7`.  

If you change this value, your results may differ from the reported novel results.



\---



\## Step 5 - Update the Evaluation Scripts to Point to the Novel Output Folder



To evaluate the novel approach, the scripts must point to the novelty output directory instead of the baseline directory.



In `replication\_scripts/evaluate\_table1\_simulation.py`, make sure the simulation path points to:



```python

SIM\_SEQ\_PATH = os.path.join(

&#x20;   BASE\_DIR,

&#x20;   "RecVAE",

&#x20;   "graphs",

&#x20;   "topk\_10",

&#x20;   "1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0\_novelty\_divrerank\_0.7",

&#x20;   "sim\_sequences.tsv"

)

```



In `replication\_scripts/evaluate\_entropy.py`, make sure the entropy path points to:



```python

ENTROPY\_PATH = os.path.join(

&#x20;   BASE\_DIR,

&#x20;   "RecVAE",

&#x20;   "graphs",

&#x20;   "topk\_10",

&#x20;   "1.0\_0.0\_gamma1\_0.0\_sigmagamma1\_0.01\_gamma2\_0.0\_sigmagamma2\_0.01\_gamma3\_0.0\_sigmagamma3\_0.01\_eta\_0.0\_novelty\_divrerank\_0.7",

&#x20;   "entropy\_metrics.tsv"

)

```



\---



\## Step 6 - Run the Same Simulation Command



After replacing `graph\_generation.py` with the novel version, run the same simulation command used for the baseline:



```bash

python handle\_modules.py "%CD%\\data\\processed\\\\" movielens-1m RecVAE generation No\_strategy False "" movielens-1m cpu 1.0 "0,0,0,0,0,0,0" "0.01,0.01,0.01,0.01,0.01,0.01,0.01" 0.0 False Horror 0.0

```



The difference is that `graph\_generation.py` now contains the novelty-enabled reranking logic, so the generated results will be written to the novelty folder.



\---



\## Step 7 - Run the Evaluation Scripts



After the simulation finishes, run:



```bash

python replication\_scripts\\evaluate\_table1\_simulation.py

python replication\_scripts\\evaluate\_entropy.py

```



These scripts should now read from the novelty folder and produce the novel results.



Expected approximate novel output:



\- \*\*HR@10 ≈ 0.3527\*\*

\- \*\*Recall@10 ≈ 0.0498\*\*

\- \*\*NDCG@10 ≈ 0.0509\*\*

\- \*\*Average TopK Entropy ≈ 0.8251\*\*

\- \*\*Average History Entropy ≈ 0.6746\*\*



\---



\## Step 8 - Switch Back to Baseline If Needed



If you want to switch back to the baseline after running the novel version, restore the original baseline `graph\_generation.py` file from the backup you created in Step 1.

