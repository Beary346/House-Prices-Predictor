import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_pipeline

df = pd.read_csv(r"C:\Users\RC\Documents\ML Coding Projects\House Prices Predictor\data\train.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.5, random_state=42
)

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

import joblib
joblib.dump(pipeline, "model.joblib")

