# Credit Card Prediction Model

Project Overview
This project involves building and evaluating machine learning models to predict credit card ownership based on various features from a banking dataset. The dataset contains information about bank customers, including their demographics, financial status, and account types.

Dataset
The dataset used in this project is Universalbank.csv, which includes the following features:

ID: Unique identifier for each customer
Age: Age of the customer
Experience: Years of experience in the banking sector
Income: Annual income of the customer
ZIP Code: Customer’s ZIP code
Family: Number of family members
CCAvg: Average credit card spending
Education: Education level of the customer (1: Undergrad, 2: Graduate, 3: Advanced)
Mortgage: Mortgage amount
Personal Loan: Whether the customer has a personal loan (1: Yes, 0: No)
Securities Account: Whether the customer has a securities account (1: Yes, 0: No)
CD Account: Whether the customer has a CD account (1: Yes, 0: No)
Online: Whether the customer uses online banking (1: Yes, 0: No)
CreditCard: Target variable indicating whether the customer has a credit card (1: Yes, 0: No)
Objective
The goal of this project is to build and evaluate machine learning models to predict whether a customer owns a credit card. The models are evaluated on their accuracy, precision, recall, f1-score, and ROC AUC score.

Methodology
Data Exploration and Preparation

Loaded and explored the dataset using pandas.
Checked for class imbalance in the target variable CreditCard.
Performed random under-sampling and over-sampling to address class imbalance.
Model Building

Split the dataset into training and testing sets.
Trained a Decision Tree Classifier on the original dataset and on the balanced datasets (under-sampled and over-sampled).
Model Evaluation

Evaluated the performance of each model using metrics such as confusion matrix, classification report, accuracy, precision, recall, f1-score, and ROC AUC score.
Results
Original Dataset

Accuracy: 61.87%
Precision: 35.23%
Recall: 40.52%
F1-score: 37.69%
ROC AUC Score: 55.35%
Over-Sampled Dataset

Accuracy: 79.93%
Precision: 75.88%
Recall: 87.48%
F1-score: 81.27%
ROC AUC Score: 79.97%
Under-Sampled Dataset

Detailed metrics to be included based on your results
Files
Universalbank.csv: The dataset used for training and evaluation.
credit_card_prediction.ipynb: Jupyter Notebook containing code for data exploration, model training, and evaluation.
Dependencies
pandas
numpy
matplotlib
scikit-learn
Installation
To run the code, make sure to install the required libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn
Usage
Place the Universalbank.csv file in the same directory as the Jupyter Notebook.
Open credit_card_prediction.ipynb in Jupyter Notebook or any compatible environment.
Run the cells in the notebook to perform data analysis, model training, and evaluation.
Conclusion
The project demonstrates the application of machine learning techniques to a real-world dataset to address class imbalance and predict credit card ownership. The models trained on the balanced datasets showed improved performance compared to the model trained on the original imbalanced dataset.

