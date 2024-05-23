# Homework8
The purpose of this project is to train a neural network model in Docker and predict the results. The data used is the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). This project follows the structure of [MarkKorvin/MLE_basic_example](https://github.com/MarkKorvin/MLE_basic_example).

## TL;DR: Execution Instructions

0. Ensure you have Docker Desktop and git installed on your machine. If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter

5. Now you should see the "Homework8" folder in your local destination. If it is there, you may close the Terminal/Powershell.

6. Open a new Terminal/Powershell at Homework8 folder

7. Build the training Docker image. Paste this code and hit enter (building may take up to 10 minutes):
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
8. Run the training container as follows:
```bash
docker run -v $(pwd)/models:/app/models training_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/models:/app/models training_image
```

9. You will see the training logs directly in the Terminal/Powershell. Once the training is complete, you will see the Classification report on the Test set, showcasing the performance of the trained model. In the local "Homework8" directory on your machine, you should also see the newly created "models" directory along with "trained_model.pth" file inside it.

10. Build the inference Docker image. Paste this code into the Terminal/Powershell and hit enter (building may take up to 10 minutes):
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=trained_model.pth --build-arg settings_name=settings.json -t inference_image .
```
11. Run the inference container as follows:
```bash
docker run -v $(pwd)/results:/app/results inference_image
```
Note: If you are using Windows Powershell, the code above may not work. In that case, try running this:
```bash
docker run -v ${PWD}/results:/app/results inference_image
```
12. After the container finishes running, you will see the inference results in the Terminal/Powershell. If you go to the "Homework8" folder, you should also see the newly created "results" directory along with "inference_results.csv" file inside it. 

## Project Structure
This project has a modular structure, where each folder serves a specific purpose. Folders "models" and "results" are not included in this repository as they are created during training and inference.

```
Homework8
├── data                      # Data files used for training and inference (uploaded and processed with load_data.py script)
│   ├── iris_inference_data.csv
│   ├── iris_train_data.csv
│   ├── load_data.py
│   └── __init__.py
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── unittests                 # Scripts used for unit tests
│   └── unittests.py
├── .gitignore                # File that filters out all the temp data, results, and models
├── README.md                 
├── __init__.py                
├── requirements.txt          # File with all the necessary libraries and their versions
├── utils.py                  # Utility functions and classes that are used in scripts
└── settings.json             # All configurable parameters and settings
```

## Prerequisites
This project requires Docker and git. An IDE (like VScode) is recommended for running unit tests or examining Python scripts. 

When running scripts outside Docker, ensure your Python environment matches the project's requirements. If using Conda, install PyTorch separately. For non-Conda environments, use `pip install -r "requirements.txt"`. 

This project uses Python 3.10.12. Different Python versions might require adjusted library versions. Consider aligning your Python version with this project's for compatibility.

## Data
The data used is the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), imported from Scikit-learn library. It contains 150 observations, 4 features and 3 labels (each having 50 entries). The data is imported and processed using the "load_data.py" script, which splits it into training and inference sets with stratification. Each set is then scaled using MinMaxScaler (fit only on training data to avoid data leakage) and saved to a .csv file. The training set has 135 rows, and the inference set contains only features (without labels) and has 15 rows. The training set is further split into train and test sets in train.py script. Before running data through training and inference, each set is converted into a Torch DataLoader with a batch size of 1 (since the datasets are small).

## Model
In this project, a neural network classifier is implemented. 

* Input Layer: 4 features
* Hidden Layer: 3 layers with 64 neurons 
* Output Layer: 3 neurons, each outputting a probability
* Activation Functions: ReLU for hidden layers and Softmax for the output layer
* Loss Function: Cross Entropy Loss coupled with Softmax function (`nn.CrossEntropyLoss`)
* Optimizer: Adam with a learning rate of 0.001 
* Number of epochs: 30 

Some parameters such as the number of neurons and layers in the hidden layers, learning rate, number of epochs, and batch size can be modified in "settings.json". 

## Training 

The initial training data ("iris_train_data.csv") is split into train and test sets with the test size being 0.2 (can be modified in "settings.json"). Once the training is complete, the model is evaluated on the test set and the classification report is displayed.

The recommended way to run the training process is in a Docker container. First, build the training image. Clone the repository to a local destination of your choice, open Terminal (on Mac) or Powershell (on Windows) at your local "Homework8" folder, and run the following code:

```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

It may take around 10 minutes to build the image, particularly because the code needs to install the packages from "requirements.txt". Once the image is built, you will see it in the "Images" tab in Docker Desktop. Then, you may run the image by pressing the run button in the application (the trained model will then be saved in the container only), but I recommend running it with this code in your Terminal:

```bash
docker run -v $(pwd)/models:/app/models training_image
```
If you are using Powershell and facing an error running the code above, try running this:
```bash
docker run -v ${PWD}/models:/app/models training_image
```
Running the container this way will create the "models" folder in the local working directory (unless it exists already) and save the trained model to it. You will need this saved model to build the inference image later. Once the container is done running, you will also see the classification report on the test set.

Alternatively, you may run the "train.py" script in your IDE. Yet again, you may run into issues when installing the required packages depending on your interpreter version.

## Inference

Before building the inference image, make sure a trained model exists locally. Go to your "Homework8" folder, find the newly created "models" folder and check if "trained_model.pth" exists in it. If the file is there, you can build the inference image by running this code in the Terminal/Powershell:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=trained_model.pth --build-arg settings_name=settings.json -t inference_image .
```

Similarly, if you want to save the inference results locally, you can do that by running the container with this code in your Terminal:
```bash
docker run -v $(pwd)/results:/app/results inference_image
```
or with this one if you are using Windows Powershell:
```bash
docker run -v ${PWD}/results:/app/results inference_image
```
This will create a "results" folder in your "Homework8" folder and add the inference results to it in a .csv file. If you don't want to save the results locally, you can just run the container by pressing the run button or typing `docker run inference_image` in your Terminal/Powershell. Regardless of whether you want to save the file or not, you will be able to see the inference results in your Terminal/Powershell once the container has finished running.

Alternatively, you may run the "run.py" script in your IDE. Yet again, you may run into issues when installing the required packages depending on your interpreter version.

Note that if you run the inference process again (whether through Docker or locally), it will overwrite the previous ouput in the "inference_results.csv" file.

## Unit Tests
To run "unittests.py", you will need to make sure you have the correct packages installed in your local environment. There are a total of 10 tests, each corresponding to a method within "train.py" and "run.py" scripts. 
