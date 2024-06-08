# Breast-Cancer-Prediction
This project focuses on predicting breast cancer using neural networks.
# Breast Cancer Prediction Using Neural Networks and Hyperparameter Tuning

## Overview
This project focuses on predicting breast cancer using neural networks. The dataset undergoes significant pre-processing, and various hyperparameters are tuned to create an accurate model. The performance of the model is evaluated using cross-validation and grid search techniques.

## Context
In this project, we aim to predict breast cancer diagnosis based on various features of cell nuclei present in the dataset. The neural network model is built, trained, and evaluated using cross-validation and grid search for hyperparameter tuning.

## Objectives:
- Develop a neural network model to predict breast cancer diagnosis.
- Pre-process the dataset to handle missing values and encode categorical features.
- Use cross-validation to evaluate the model's performance.
- Perform hyperparameter tuning using grid search to optimize the model.

## Steps Involved:

### 1. Data Pre-processing:
- Load the dataset using Pandas.
- Handle missing values and encode categorical variables if necessary.

### 2. Building the Neural Network:
- Create a neural network using Keras with three layers:
  - Input layer with 16 units and ReLU activation.
  - Hidden layer with 16 units and ReLU activation.
  - Output layer with 1 unit and sigmoid activation.
- Apply dropout regularization to prevent overfitting.

### 3. Model Training and Evaluation:
- Train the model using Keras' `KerasClassifier` with 100 epochs and a batch size of 10.
- Evaluate the model using cross-validation (`cross_val_score` with 10 folds) and calculate the mean and standard deviation of accuracy.

### 4. Hyperparameter Tuning:
- Use `GridSearchCV` to perform hyperparameter tuning with parameters such as batch size, number of epochs, optimizer, loss function, kernel initializer, activation function, and number of neurons.
- Identify the best hyperparameters and evaluate the model's accuracy.

## Files:

- **breast_cancer_cross.py**: Contains the code for data pre-processing, model training, and cross-validation.
- **breast_cancer_tuning.py**: Contains the code for hyperparameter tuning using grid search.
- **classificador_breast.json**: JSON file containing the model architecture.
- **entradas-breast.csv**: CSV file containing the input features of the dataset.
- **saidas-breast.csv**: CSV file containing the output labels of the dataset.

## How to Run:
1. Clone the repository:
    ```sh
    git clone <repository URL>
    ```

2. Navigate to the project directory:
    ```sh
    cd <repository name>
    ```

3. Ensure you have the required libraries installed. You can install them using:
    ```sh
    pip install pandas keras scikit-learn
    ```

4. Run the scripts to see the results:
    ```sh
    python breast_cancer_cross.py
    python breast_cancer_tuning.py
    ```

## Contribution
Contributions are welcome! Feel free to open issues and submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
