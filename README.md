**Heart Disease Prediction - Machine Learning Project**
## 🖼️ Project Workflow
![Heart Disease Prediction](image/image.jpg)

# Heart Disease Prediction - Machine Learning Project


📌 Overview

This project focuses on predicting the likelihood of heart disease using the UCI Heart Disease dataset.
The workflow is divided into 5 days, covering everything from setup to model evaluation and future improvements.

The main objective is to build and evaluate multiple machine learning models that can assist in early diagnosis of heart disease.

📂 Dataset

Source: Heart Disease Dataset (Kaggle)

File Used: heart_disease_uci.csv

Features: Patient health metrics such as age, sex, chest pain type, cholesterol, blood pressure, etc.

Target: Presence or absence of heart disease (binary classification)

⚙️ Technologies Used

Python (Google Colab environment)

Libraries:

pandas, numpy – data handling

matplotlib, seaborn – visualization

scikit-learn – ML models & evaluation

joblib – model persistence

📅 Project Roadmap (5 Days)
Day 1 – Setup & Data Collection

Configured Google Colab environment

Connected Kaggle API and downloaded dataset

Imported libraries (pandas, matplotlib, seaborn, sklearn)

Loaded dataset into pandas DataFrame

Day 2 – Data Preprocessing

Checked for missing/null values

Encoded categorical variables

Normalized/standardized numerical features

Performed train-test split for modeling

Day 3 – Exploratory Data Analysis (EDA)

Visualized data distributions (histograms, bar plots)

Correlation heatmap to find feature relationships

Compared heart disease vs. non-heart disease samples

Identified most important health indicators

Day 4 – Model Training

Implemented machine learning models:

Logistic Regression

Random Forest Classifier

Gradient Boosting / other ensemble methods

Tuned hyperparameters with GridSearchCV/RandomizedSearchCV

Saved trained models using joblib

Day 5 – Model Evaluation & Results

Evaluated models using:

Accuracy, Precision, Recall, F1-score

Confusion Matrix

ROC-AUC curve

Compared results of all models

Documented findings and identified best-performing model

Suggested future improvements

🚀 Usage
1. Clone Repository
git clone https://github.com/AnushkaSavita/Disease-prediction-model.git
cd Disease-prediction-model

2. Open in Google Colab

Click the badge below to open and run in Colab:

3. Run Notebook

Upload your kaggle.json API key

Execute cells day by day or all at once

📈 Results (Sample)
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	~0.85	~0.83	~0.82	~0.82
Random Forest	~0.88	~0.87	~0.85	~0.86
Gradient Boosting	~0.90	~0.89	~0.88	~0.89

(Values are approximate — actual scores in notebook.)

🔮 Future Improvements

Try deep learning models with TensorFlow/PyTorch

Improve feature engineering with domain knowledge

Deploy trained model as a Flask/Django web app

Integrate with real-time health data for predictive healthcare

👩‍💻 Author

Anushka Savita
B.Tech CSE (AI Specialization)
