# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# Exploratory Data Analysis (EDA)
# Checking for missing values
print("Training set missing values:\n", train_df.isnull().sum())
print("Test set missing values:\n", test_df.isnull().sum())

# Visualize distributions
plt.figure(figsize=(10, 6))
train_df['Age'].hist(bins=30)
plt.title("Age Distribution in Train Set")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Relationship between features and survival
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title("Survival by Passenger Class")
plt.show()

# Data Preprocessing
# Impute missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop 'Cabin' column due to high missing values
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)

# Encoding categorical variables
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Feature and Target Selection
X = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = train_df['Survived']
X_test = test_df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Split the training data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# Initialize LightGBM and XGBoost
lgbm = lgb.LGBMClassifier()
xgbm = xgb.XGBClassifier(eval_metric='logloss')

# Train LightGBM
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_val)

# Train XGBoost
xgbm.fit(X_train, y_train)
y_pred_xgbm = xgbm.predict(X_val)

# Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("-" * 30)

evaluate_model(y_val, y_pred_lgbm, "LightGBM")
evaluate_model(y_val, y_pred_xgbm, "XGBoost")

# Hyperparameter Tuning (optional)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# Using GridSearchCV for tuning LightGBM
grid_lgbm = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=5, scoring='accuracy')
grid_lgbm.fit(X_train, y_train)
best_lgbm = grid_lgbm.best_estimator_
y_pred_lgbm_tuned = best_lgbm.predict(X_val)

# Using GridSearchCV for tuning XGBoost
grid_xgbm = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss'), param_grid, cv=5, scoring='accuracy')
grid_xgbm.fit(X_train, y_train)
best_xgbm = grid_xgbm.best_estimator_
y_pred_xgbm_tuned = best_xgbm.predict(X_val)

# Evaluate Tuned Models
evaluate_model(y_val, y_pred_lgbm_tuned, "Tuned LightGBM")
evaluate_model(y_val, y_pred_xgbm_tuned, "Tuned XGBoost")

# Final Prediction on Test Set
final_predictions_lgbm = best_lgbm.predict(X_test)
final_predictions_xgbm = best_xgbm.predict(X_test)

# Print final predictions
print("Final LightGBM predictions on test set:\n", final_predictions_lgbm)
print("Final XGBoost predictions on test set:\n", final_predictions_xgbm)
