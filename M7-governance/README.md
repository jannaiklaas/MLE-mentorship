# Homework 7: Data Governance with Data Version Control (DVC)

## TL;DR: Execution Instructions

0. Ensure you have git and dvc installed on your machine. If you know how to clone a repository and have already done so with this project, jump to Step 5. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter
5. Now you should see the `MLE-mentroship` folder in your local destination. If it is there, run the following command in the Terminal obtian the data from the remote storage.
```bash
cd M7-governance
dvc pull
```
6. Reproduce the preprocessing and training pipeline (see `dvc.yaml`) by running the following command:
```bash
dvc repro
```
7. You can check the validation metrics by running
```bash
dvc metrics show
```
8. In the coding environment or text editor of your choice, modify the `preprocess_data.py` script. As a suggestion, in Line 139, you may change `use_lemmatization` to False and/or change `vectorization_type` to "tf-idf". Once you modify the file, save your changes.
9. Run the following commands to see the change in metrics:
```bash
dvc repro && dvc metrics diff
```
