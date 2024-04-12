# Homework 3: Experiment Tracking

## Experiments

There are five different experiments conducted, each with a different setup:
1. `LogisticRegression` model without any features dropped (`feature_set_version = 1`);
2. `LinearSVC` model with three features dropped, namely: `SkinThickness`, `BloodPressure` and `Insulin` (`feature_set_version = 2`);
3. `LogisticRegression` model more features dropped, namely `DiabetesPedigreeFunction` and `BMI` on top of the features dropped in Experiment 2 (`feature_set_version = 3`);
4. `LinearSVC` model without any features dropped (`feature_set_version = 1`);
5. `XGBoostClassifier` model without any features dropped (`feature_set_version = 1`);

Moreover, each experiment went through hyperparameter tuning using Optuna with child runs. 

More experiments can be created, as long as the script is named `train.py` and located inside `experiment_{your_experiment_number}` folder. Make sure to replace `your_experiment_number` with a number of your choice.


## Execution Instructions

0. Ensure you have [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine (in case you don't, click the corresponding hyperlinks). If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the `MLE-mentorship` folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at "M3-experiment-tracking" folder (inside the "MLE-mentorship" folder). This will be your working directory.

7. To start a server and run an experiment, start Docker Engine and run the following command:
```bash
EXPERIMENT_NUMBER={your_experiment_number} docker-compose up --build
```
Make sure to replace `your_experiment_number` with the experiment number you want to run.

8. Once the MLflow server is up, you should be able to access it through your web browser by navigating to `http://localhost:5001`.

9. The experiment's artifacts are locally saved at `mlflow/artifacts/` . You should be able to see in the `MLE-mentorship` folder after the client container has finished running.