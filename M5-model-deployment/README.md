# Homework 5: Deploying the Model with Brain Tumor Prediction
The dataset contains 253 brain MRI images for brain tumor detection. The solution implemented in this project is a slightly modified version of the solution originally available on [Kaggle](https://www.kaggle.com/code/seifwael123/brain-tumor-detection-cnn-vgg16?rvi=1).
## mle-hw5-package
Except for tests, all project codes and documentation are available on [TestPyPI](https://test.pypi.org/project/mle-hw5-package/). To download and install it run the following command in your Terminal/Powershell:
```bash
pip install -i https://test.pypi.org/simple/ mle-hw5-package==0.0.9
```
You will not need the package to run inference provided you can clone project's repository on [GitHub](https://github.com/jannaiklaas/MLE-mentorship/tree/main/M5-model-deployment). You will only need the package to run code tests.

Note: the package does **not** include any datasets, neither it includes the pre-trained model. These can be obtained by running `data_loader.py` and `train.py` in this specific order. 

## Getting Started
0. If you know how to clone a repository and have already done so with this project, jump to Step 6. 
1. Choose a local destination for the project, e.g. a new folder on your Desktop
2. Open Terminal (if you are on Mac) or Powershell (if you are on Windows) at the chosen destination
3. Copy the repository's URL from GitHub
4. In the Terminal/Powershell, type `git clone`, paste the URL you just copied and hit enter
5. Now you should see the `MLE-mentorship` folder in your local destination. If it is there, you may close the Terminal/Powershell.
6. You will need data to run inference. Navigate to `MLE-mentorship`. Run the following command in your Terminal/Powershell at the project directory named `M5-model-deployment`.
```bash
# Assuming you are in M5-model-deployment
python3 ./src/mle-hw5-package/data_loader.py
```
7. Now you will need the model, with which you will run the inference.