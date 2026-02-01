import pandas as pd
import joblib
from sklearn.metrics import classification_report

test = pd.read_csv(r"C:\Users\Ryan Craft\Documents\ML Coding Projects\House Prices Predictor\data\test.csv")
X_test = test

test_id = test["Id"]

pipeline = joblib.load("model.joblib")
y_pred = pipeline.predict(X_test)

with open("submission.csv", mode='w') as submissionfile:
    print("Id,SalePrice\n")
    for i in range(len(test)):
        submissionfile.write(f"{test_id[i]}, {y_pred[i]}\n")
        