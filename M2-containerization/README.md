# Homework 2: Containerization
The dataset contains 253 brain MRI images for brain tumor detection. The solution implemented in this project is a slightly modified version of the solution originally available on [Kaggle](https://www.kaggle.com/code/seifwael123/brain-tumor-detection-cnn-vgg16?rvi=1).
## Execution Instructions

0. Ensure you have [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine (in case you don't, click the corresponding hyperlinks). If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the "MLE-mentorship" folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at "M2-containerization" folder (inside the "MLE-mentorship" folder). This will be your working directory.

7. You will need a few modules to load the data from the remote server, namely `scikit_learn`, `opencv-python`, and `requests`. If you don't have those modules installed, you can install them from the "requirements.txt" file by running the following command:

```bash
pip install -r "requirements.txt"
```

8. Download the data by running this code in Terminal/Powershell:
```bash
python3 src/data_loader.py
```
* Alternatively, you can run the script "data_loader.py" interactively in the VSCode IDE by pressing the run button.

* Once you run the script, you will see logs directly in your Terminal/Powershell. Once the data is dowloaded, make sure the "data" directory is created inside the "M2-containerization" folder. Go to the next step only after ensuring the "data"folder exists and is populated. 

9. Open Docker Desktop. Build the training Docker image. Paste this code in the Terminal/Powershell and hit enter (it may take up to 5 min to download all the packages):
```bash
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f ./src/train/Dockerfile -t training_image .
```
10. Run the training container as follows (execution may take up to 10 minutes):
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results training_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/models:/app/models -v ${PWD}/results:/app/results training_image
```

11. You will see the training logs directly in the Terminal/Powershell. Once the training is complete, model's validation metrics will be displayed in the terminal, which you can also check in the "metrics.txt" file. The file will be located in the newly created "results" directory. In the working directory, there will also be a newly created "models" directory, containing the trained model.

12. Build the inference Docker image. Paste this code into the Terminal/Powershell and hit enter:
```bash
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f ./src/inference/Dockerfile -t inference_image .
```
13. Run the inference container as follows:
```bash
docker run -v $(pwd)/results:/app/results inference_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/results:/app/results inference_image
```
14. After the container finishes running, you will see the inference metrics in the Terminal/Powershell, which will be automatically added to the "metrics.txt" file.

