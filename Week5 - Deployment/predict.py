import pickle

model_file = "model_C=1.0.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

# Test prediction on sample customer
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 5,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# Predict churn probability for sample customer
X = dv.transform([customer])
y_pred = model.predict_proba(X)[0,1]

print(f"Input: {customer}")
print(f"Churn probability: {round(y_pred, 3)*100}%")