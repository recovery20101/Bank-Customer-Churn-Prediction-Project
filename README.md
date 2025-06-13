# Bank Customer Churn Prediction Project
This project focuses on developing a machine learning model to predict bank customer churn. The main goal is to identify customers who are likely to churn so the bank can take proactive measures to retain them. The project includes stages of exploratory data analysis (EDA), data preprocessing, development and comparison of various machine learning models, hyperparameter tuning, and simulation of model deployment in a web application.

## Problem Solved
Banks frequently face customer churn, leading to significant financial losses. Churn prediction allows for:

* Timely identification of at-risk customers.
* Development of targeted retention strategies.
* Optimization of marketing campaigns.

## Key Components and Functionality
* Exploratory Data Analysis (EDA): In-depth data exploration to uncover patterns, distributions, and relationships between features and customer churn.
* Data Preprocessing: Data cleaning, handling categorical features (One-Hot Encoding), scaling numerical features, and creating new features (e.g., HasNoBalance).
* Modeling: Development and evaluation of predictive models, including Logistic Regression, Random Forest, and XGBoost.
* Model Optimization: Using GridSearchCV and RandomizedSearchCV for fine-tuning the hyperparameters of the best model (XGBoost).
* Best Model Selection: XGBoost was chosen as the best model based on Precision, Recall, F1-Score, and AUC-ROC metrics for the churn class.
* Deployment Simulation: A simple Flask web application to demonstrate the model's operation in a "production" environment, where users can input customer data and receive a churn prediction.

## Dataset
The Churn_Modelling.csv dataset was used, containing information on 10,000 bank customers and including the following features:

* CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
* Target variable: Exited (1 - churned, 0 - remained).

## Technology Stack
* Python 3.13
* Core Libraries:
    * pandas (for data manipulation)
    * numpy (for numerical operations)
    * scikit-learn (for data preprocessing and models)
    * xgboost (for modeling)
    * matplotlib, seaborn (for data visualization)
    * joblib (for saving/loading models)
    * Flask (for the web application)
  
## Installation and Running the Project
To run this project locally, follow these steps:

1. Clone the repository:

```
git clone <Your Repository URL>
cd <Your Repository Name>
```
2. Create and activate a virtual environment (recommended):

```
python -m venv venv
# For Windows:
.\venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```
3. Install dependencies:

```
pip install -r requirements.txt
```
4. Download the data:
Ensure that the Churn_Modelling.csv file is located in the data/ folder.

5. Train and save the model:
Run the script that trains the model and saves it, along with preprocessing objects:

```
python src/train_and_save_model.py
```
This step will create (or update) the xgb_model.pkl, preprocessor.pkl, final_features_order.pkl files in the trained_models/ folder.

6. Run the Flask web application:

```
cd web_app
python app.py
```
Open your browser and navigate to the address displayed in the terminal (usually http://127.0.0.1:5000/).

## Project Structure
```
.
├── data/
│   └── Churn_Modelling.csv         # Raw data
├── notebooks/
│   └── Churn_Prediction_EDA_and_Modeling.ipynb # Jupyter Notebook for EDA and experiments
├── src/
│   └── train_and_save_model.py     # Script for training and saving the final model and preprocessor
├── trained_models/                 # Saved model and preprocessing objects
│   ├── xgb_model.pkl
│   ├── preprocessor.pkl
│   └── final_features_order.pkl
├── web_app/
│   ├── app.py                      # Main Flask application file
│   └── templates/
│       └── index.html              # HTML template for the web interface
├── requirements.txt                # List of Python dependencies
└── README.md                       # This file
```

## Results and Conclusions
* Best Model: XGBoost showed the best results, achieving the following metrics on the test set:
    * AUC-ROC: ~0.866
    * Recall (for churn class): ~0.75
    * F1-Score (for churn class): ~0.61
* Key Churn Factors (based on EDA and analysis of coefficients/feature importance):
    * Customers with low or zero balance.
    * Geographical location (e.g., customers from Germany may have a higher churn risk).
    * Inactive members.
    * Customer age.

## Future Improvements (Optional)
* Add more feature engineering (e.g., interactions between features).
* Experiment with other models (e.g., LightGBM, CatBoost).
* Extend the web application (add charts, admin panel, database).
* Implement automated model retraining.
* Perform real deployment using Docker and cloud services (AWS, Azure, GCP).
