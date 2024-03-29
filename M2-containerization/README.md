# Homework 2: Containerization
## TL;DR: Execution Instructions

0. Ensure you have [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your machine (in case you don't, click the corresponding hyperlinks). If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the "data-science-task" folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at "data-science-task" folder

7. You will need a few modules to load the data from the remote server, namely `scikit_learn`, `opencv-python`, and `requests`. If you don't have those modules installed, you can install them from the "requirements.txt" file by running the following command:

```bash
pip install -r "requirements.txt"
```

8. Download the data by running this code in Terminal/Powershell:
```bash
python3 src/data_loader.py
```
* Alternatively, you can run the script "data_loader.py" interactively in the VSCode IDE by pressing the run button.

* Once you run the script, you will see logs directly in your Terminal/Powershell. Once the data is dowloaded, make sure the "data" directory is created inside the "data-science-task" folder. Click on "data" folder -> "raw" -> "train" or "inference" and check if "train.csv" and "test.csv" exist at the respective locations. Go to the next step only after ensuring both datasets exist. 

9. Open Docker Desktop. Build the training Docker image. Paste this code in the Terminal/Powershell and hit enter:
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

11. You will see the training logs directly in the Terminal/Powershell. Once the training is complete, model's validation metrics will be displayed in the terminal, which you can also check in the "metrics.txt" file. The file will be located in the newly created "outputs" directory, inside "predictions" folder. The "outputs" directory will also contain "models", "figures" and "processors" folders. Each of these folders should have relevant files stored. For example, the "figures" folder will have two .png files: one with a feature importance plot and one with a validaiton confusion matrix.

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
14. After the container finishes running, you will see the inference metrics in the Terminal/Powershell, which will be automatically added to the "metrics.txt" file. In the local "data-science-task" folder, you should also see the newly created "predictions.csv" file inside "predictions" folder in "outputs" directory.  

## Project Structure
This project has a modular structure, where each folder serves a specific purpose. Folders "data" and "outputs" are not included in this repository as they are created during training and inference.

This is the original structure of the repository:
```
/data-science-task/
├── notebooks                 # Notebooks containing the Data Science part
│   └── Iklaas_J_Final_Project_DS23.ipynb              
├── src                       # All necesary scripts and Dockerfiles
│   ├── inference             # Scripts and Dockerfiles used for inference
│   │   ├── Dockerfile
│   │   ├── run_inference.py
│   │   └── __init__.py
│   ├── train                 # Scripts and Dockerfiles used for training
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   └── __init__.py
│   ├── data_loader.py        # Script to download raw data
│   ├── text_processor.py     # Classes and methods to preprocess text
│   └── __init__.py
├── .gitignore                # File that filters out all data and outputs
├── README.md                        
└── requirements.txt          # File with all the necessary libraries and their versions
```
<a id="data-creation"></a> 
After running "data_loader.py", "data" folder should appear in the project's directory :
```
/data-science-task/
├── data
│   └── raw                   # Raw data directory
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
├── notebooks
│   └── Iklaas_J_Final_Project_DS23.ipynb              
<...>                      
└── requirements.txt     
```
<a id="post-train"></a>
After running "train.py" either locally or in Docker, processed data will be added to "data" directory, and "outputs" folder will also appear:
```
/data-science-task/
├── data
│   └── raw
│   │   ├── inference
│   │   │   └── test.csv
│   │   └── train
│   │       └── train.csv
│   └── processed             # Processed data directory
│       └── train
│           ├── train_processed.csv
│           └── validation_processed.csv
<...>
├── outputs                   # Directory for training and inference outputs
│   ├── figures               # Directory for validation and inference plots
│   │   ├── feature_importance.png
│   │   └── model_1_validation_confusion_matrix.png
│   ├── models                # Directory for trained models
│   │   └── model_1.pkl
│   ├── predictions           # Directory for inference predictions
│   │   └── metrics.txt       # File with validation and inference metrics
│   └── processors            # Directory for fit text processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```
<a id="post-inference"></a>
Finally, after running inference, the project directory will look as follows:
```
/data-science-task/
├── data
│   <...>
│   └── processed
│       ├──inference
│       │   └── test_processed.csv
│       └── train
│           ├── train_processed.csv
│           └── validation_processed.csv
<...>
├── outputs
│   ├── figures
│   │   ├── feature_importance.png
│   │   ├── model_1_inference_confusion_matrix.png
│   │   └── model_1_validation_confusion_matrix.png
│   ├── models
│   │   └── model_1.pkl
│   ├── predictions
│   │   ├── metrics.txt         # Inference metrics will be added to this file
│   │   └── predictions.csv     # Original inference with predictions added
│   └── processors
│       └── processor_1.pkl
<...>                      
└── requirements.txt      
```

## Prerequisites
This project requires installed [Python](https://www.python.org/downloads/) (preferably version 3.10.12), [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), and [Docker Desktop](https://www.docker.com/products/docker-desktop/). An IDE (like [VScode](https://code.visualstudio.com/)) is recommended for running or examining Python scripts. Alternatively, they can be run in Terminal/Powershell. 

When running scripts outside Docker, ensure your Python environment matches the project's requirements. If using Conda, install libraries separately. For non-Conda environments, use `pip install -r "requirements.txt"`. 

This project uses Python 3.10.12. Different Python versions might require adjusted library versions. Consider aligning your Python version with this project's for compatibility.

### Forking and Cloning from GitHub
To start using this project, you first need to create a copy on your own GitHub account by 'forking' it. On the main page of the `data-science-task` project, click on the 'Fork' button at the top right corner. This will create a copy of the project under your own account. You can then 'clone' it to your local machine for personal use. To do this, click the 'Code' button on your forked repository, copy the provided link, and use the `git clone` command in your terminal followed by the copied link. This will create a local copy of the repository on your machine, and you're ready to start!

### Setting Up Development Environment
Next, you need to set up a suitable Integrated Development Environment (IDE). Visual Studio Code (VSCode) is a great tool for this. You can download it from the official website (https://code.visualstudio.com/Download). After installing VSCode, open it and navigate to the `File` menu and click `Add Folder to Workspace`. Navigate to the directory where you cloned the forked repository and add it. For running scripts, open a new terminal in VSCode by selecting `Terminal -> New Terminal`. Now you can execute your Python scripts directly in the terminal.

### Installing Docker Desktop
Installing Docker Desktop is a straightforward process. Head over to the Docker official website's download page ([Docker Download Page](https://www.docker.com/products/docker-desktop)), and select the version for your operating system - Docker Desktop is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. 

Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. It will typically show up in your applications or programs list. After launching, Docker Desktop will be idle until you run Docker commands. This application effectively wraps the Docker command line and simplifies many operations for you, making it easier to manage containers, images, and networks directly from your desktop. 

Keep in mind that Docker requires you to have virtualization enabled in your system's BIOS settings. If you encounter issues, please verify your virtualization settings, or refer to Docker's installation troubleshooting guide. Now you're prepared to work with Dockerized applications!

## Data
The raw data is not included with the project by default. It should downloaded from an [online source](https://github.com/jannaiklaas/datasets/tree/main/movie-reviews) or uploaded from the local machine. The way the data is uploaded is by running "data_loader.py" script. This script utilizes `requests` module, and here's how you can install it if you do not have it installed yet. Once you have the remote repository cloned to your local destination, you can run the following code in the Terminal/Powershell at the local "data-science-task" repository's location:

```bash
pip install requests
```
Alternatively you can install it from the "requirements.txt" file:
```bash
pip install -r "requirements.txt"
```
If you are using Anaconda, then you can download the module by running this code:
```bash
conda install requests
```
Once you have the module installed, you can run the "data_loader.py" script. One way to do it is by running this code in your Terminal/Powershell:
```bash
python3 src/data_loader.py
```
You can also run the script interactively in the VSCode IDE by pressing the run button.

If the data is already on your local machine, you can provide the paths to the raw data :

```bash
python3 src/data_loader.py --local_train_path /path_to_local_train.csv --local_test_path /path_to_local_test.csv
```

Replace `path_to_local_train.csv` and `/path_to_local_test.csv` with actual paths on your local machine where your raw datasets are located.

Running the "data_loader.py" script will create "data" folder at the local "data-science-task" location. Inside you should see the "train.csv" and "test.csv" files, as shown in the [project structure](#data-creation).

## Training 

The recommended way to run the training process is in a Docker container. First, build the training image. Open Docker Desktop to start the Docker engine. In the Terminal/Powershell at your local "data-science-task" folder run the following code:

```bash
docker build -f ./src/train/Dockerfile -t training_image .
```
Once the image is built, you will see it in the "Images" tab in Docker Desktop. Then, you may run the image by pressing the run button in the application (the outputs will then be saved in the container only), but I recommend running it with this code in your Terminal (execution may take up to 10 minutes):

```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data training_image
```
If you are using Powershell and facing an error running the code above, try running this:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data training_image
```
Running the container this way will create the "outputs" folder in the local working directory (unless it exists already) and save the trained model, fit preprocessor, test (validation) metrics and plots to it. It will also save preprocessed data to the "data" folder, as shown in the [project structure](#post-train). The saved trained model and preprocessor will be used to build the inference image later. Once the container is done running, you will also see the performance metrics on the test set in the coding environement.

Alternatively, you may run the "train.py" script in your IDE as follows:

```bash
python3 src/train/train.py
```

Yet again, if you choose to run it outside Docker, you may encounter issues when installing the required packages depending on your interpreter version.

## Inference

Before building the inference image, make sure a trained model exists locally. Go to your "data-science-task" folder -> "outputs" -> "models" and check if "model_1.pkl" exists in it. Also check if "processor_1.pkl" exists at "data-science-task" -> "outputs" -> "processors". If both files exist, you can build the inference image by running this code in the Terminal/Powershell:
```bash
docker build -f ./src/inference/Dockerfile --build-arg model_name=model_1.pkl --build-arg processor_name=processor_1.pkl -t inference_image .
```

Similarly, if you want to save the inference results locally, you can do that by running the container with this code in your Terminal:
```bash
docker run -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data inference_image
```
or with this one if you are using Windows Powershell:
```bash
docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data inference_image
```
Once the container is done running, the inference predictions will be saved to "predictions.csv" at "outputs"/"predictions". The inference metrics will be displayed in the Terminal/Powershell and also added to "metrics.txt" file (which should already contain validation metrics). To see the complete list of outputs and newly created files, check the [project structure](#post-inference).

Alternatively, you may run the "run_inference.py" script in your IDE as follows:

```bash
python3 src/inference/run_inference.py
```

Yet again, if you choose to run it outside Docker, you may encounter issues when installing the required packages depending on your interpreter version.


