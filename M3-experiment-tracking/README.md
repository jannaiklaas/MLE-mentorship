# Homework 3: Experiment Tracking
## Execution Instructions

0. Ensure you have [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine (in case you don't, click the corresponding hyperlinks). If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the "MLE-mentorship" folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at "M3-experiment-tracking" folder (inside the "MLE-mentorship" folder). This will be your working directory.

7. Open Docker Desktop. Build the Docker images. Paste this code in the Terminal/Powershell and hit enter (it may take up to 5 min to download all the packages):
```bash
docker-compose build --no-cache
```
8. To start the server and the experiments, run::
```bash
docker-compose up
```
9. Once the MLflow server is up, you should be able to access it through your web browser by navigating to `http://localhost:5001`.
