# Credit Card Prediction Model

## 1. Project Overview
This project involves building and evaluating machine learning models to predict credit card ownership based on various features from a banking dataset. The dataset contains information about bank customers, including their demographics, financial status, and account types.

## 2. Dataset
The dataset used in this project is `Universalbank.csv`, which includes the following features:

| Feature            | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `ID`               | Unique identifier for each customer                              |
| `Age`              | Age of the customer                                              |
| `Experience`       | Years of experience in the banking sector                        |
| `Income`           | Annual income of the customer                                    |
| `ZIP Code`         | Customer’s ZIP code                                              |
| `Family`           | Number of family members                                         |
| `CCAvg`            | Average credit card spending                                     |
| `Education`        | Education level of the customer (1: Undergrad, 2: Graduate, 3: Advanced) |
| `Mortgage`         | Mortgage amount                                                  |
| `Personal Loan`    | Whether the customer has a personal loan (1: Yes, 0: No)         |
| `Securities Account` | Whether the customer has a securities account (1: Yes, 0: No)   |
| `CD Account`       | Whether the customer has a CD account (1: Yes, 0: No)            |
| `Online`           | Whether the customer uses online banking (1: Yes, 0: No)         |
| `CreditCard`       | Target variable indicating whether the customer has a credit card (1: Yes, 0: No) |

## 3. Objective
The goal of this project is to build and evaluate machine learning models to predict whether a customer owns a credit card. The models are evaluated based on accuracy, precision, recall, f1-score, and ROC AUC score.

## 4. Methodology

### 4.1 Data Exploration and Preparation
- Loaded and explored the dataset using pandas.
- Checked for class imbalance in the target variable `CreditCard`.
- Performed random under-sampling and over-sampling to address class imbalance.

### 4.2 Model Building
- Split the dataset into training and testing sets.
- Trained a **Decision Tree Classifier** on the original dataset and on the balanced datasets (under-sampled and over-sampled).

### 4.3 Model Evaluation
- Evaluated the performance of each model using metrics such as:
  - Confusion matrix
  - Classification report
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC Score

## 5. Results

| Dataset            | Accuracy | Precision | Recall | F1-Score | ROC AUC Score |
|--------------------|----------|-----------|--------|----------|---------------|
| Original Dataset   | 61.87%   | 35.23%    | 40.52% | 37.69%   | 55.35%        |
| Over-Sampled Dataset | 79.93%  | 75.88%    | 87.48% | 81.27%   | 79.97%        |
| Under-Sampled Dataset | [To be added based on results] | [To be added] | [To be added] | [To be added] | [To be added] |

## 6. Files
- `Universalbank.csv`: The dataset used for training and evaluation.
- `credit_card_prediction.ipynb`: Jupyter Notebook containing code for data exploration, model training, and evaluation.

## 7. Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## 8. Installation
To run the code, make sure to install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## 9. Usage
1. Place the `Universalbank.csv` file in the same directory as the Jupyter Notebook.
2. Open `credit_card_prediction.ipynb` in Jupyter Notebook or any compatible environment.
3. Run the cells in the notebook to perform data analysis, model training, and evaluation.

## 10. Conclusion
This project demonstrates the application of machine learning techniques to a real-world dataset to address class imbalance and predict credit card ownership. The models trained on the balanced datasets showed improved performance compared to the model trained on the original imbalanced dataset.
```
