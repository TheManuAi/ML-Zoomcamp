# Import required libraries

import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Cross-validation parameters
n_splits = 5
C = 1.0
output_file = f"model_C={C}.bin"

# Load and preprocess data
df = pd.read_csv("Churn.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Normalize categorical values
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Clean totalcharges column
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# Convert churn to binary
df.churn = (df.churn == 'yes').astype(int)

# Split data into train and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Define feature columns
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# Train logistic regression model
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model

# Make predictions on validation set
def predict(df_val, dv, model):
    dicts = df_val[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Perform K-Fold cross-validation
print(f"Doing validation with C = {C}")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"Auc on fold {fold} is {auc}")
    fold += 1

print("\nValidation results:")
# Print cross-validation results
print("C=%s %.3f +-%.3f" % (C, np.mean(scores), np.std(scores)))

# Train final model on full training set
print("\nTraining the final model")

dv, model = train(df_full_train, df_full_train.churn.values, C=1)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

# Evaluate final model
auc = roc_auc_score(y_test, y_pred)
print(f"Auc = {auc}")

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"\nThe model is saved in {output_file}")