import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load data from the CSV file.
df = pd.read_csv('../data/Churn_Modelling.csv')

# Remove columns that do not provide informative value for modeling.
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Split data into features (X) and target variable (y).
X = df.drop('Exited', axis=1)
y = df['Exited']

# Create a new feature 'HasNoBalance', indicating if the balance is zero.
X['HasNoBalance'] = (X['Balance'] == 0).astype(int)

# Defining characteristics for ColumnTransformer


# Numerical features.
features_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Categorical features.
features_to_onehot = ['Geography', 'Gender']

# Binary features to be passed through unchanged.
features_passthrough = ['HasCrCard', 'IsActiveMember', 'HasNoBalance']

# Check: ensure all columns from X are covered.
all_defined_features = set(features_to_scale + features_to_onehot + features_passthrough)
all_X_cols = set(X.columns.tolist())

if all_defined_features != all_X_cols:
    print("WARNING: Not all columns from X were included in the transformation or there are unknown columns!")
    print("X columns that were NOT defined:", all_X_cols - all_defined_features)
    print("Columns defined but missing in X:", all_defined_features - all_X_cols)


# Creating a ColumnTransformer
# Create a ColumnTransformer for feature preprocessing.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_to_scale),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_to_onehot),
        ('binary_passthrough', 'passthrough', features_passthrough)
    ],
    remainder='drop'
)

# Getting the final column names after transformation
dummy_X_transformed = preprocessor.fit_transform(X.head()) # Fit on a small subset
scaled_features_names = features_to_scale
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(features_to_onehot)
passthrough_features_names_actual = preprocessor.named_transformers_['binary_passthrough'].get_feature_names_out(features_passthrough)

final_columns_order = []
final_columns_order.extend(scaled_features_names)
final_columns_order.extend(ohe_feature_names)
final_columns_order.extend(passthrough_features_names_actual)


# TRAINING THE XGBOOST MODEL

# Use the previously found best parameters for XGBoost.
best_xgb_params = {'colsample_bytree': 0.7, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.8}

# Calculate scale_pos_weight for class balancing, using the ratio of negative to positive classes.
scale_pos_weight_value = y.value_counts()[0] / y.value_counts()[1]

# Initialize the XGBoost Classifier model.
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic',
                                   eval_metric='logloss',
                                   scale_pos_weight=scale_pos_weight_value,
                                   random_state=42,
                                   **best_xgb_params)

# Create the Pipeline
print("Creating and training the machine learning pipeline...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # First step: apply preprocessing
    ('classifier', xgb_classifier)  # Second step: train the classifier
])

# Train the entire pipeline on the full dataset.
model_pipeline.fit(X, y)
print("Pipeline training completed.")

# Save objects for production
# Create a directory for saving models if it doesn't exist.
output_dir = '../trained_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the entire trained pipeline.
joblib.dump(model_pipeline, os.path.join(output_dir, 'churn_prediction_pipeline.pkl'))
print(f"Machine learning pipeline saved to {os.path.join(output_dir, 'churn_prediction_pipeline.pkl')}")

# Save the order and names of the final features.
joblib.dump(final_columns_order, os.path.join(output_dir, 'final_features_order.pkl'))
print(f"Final features order saved to {os.path.join(output_dir, 'final_features_order.pkl')}")
