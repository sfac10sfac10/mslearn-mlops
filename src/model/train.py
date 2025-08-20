# Import libraries

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import mlflow
import mlflow.sklearn


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog(log_models=False)

    with mlflow.start_run():
        # read data
        df = get_csvs_df(args.training_data)

        # split data
        X_train, X_test, y_train, y_test = split_data(df)

        # train model
        model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

        # Log the model **with conda.yaml**
        conda_env = mlflow.sklearn.get_default_conda_env()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=conda_env
        )


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df):
    feature_cols = [
        'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
        'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'
    ]
    X = df[feature_cols].values
    y = df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=0)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    # Guardar artefactos
    os.makedirs("outputs", exist_ok=True)
    import joblib
    joblib.dump(model, "outputs/model.joblib")
    with open("outputs/metrics.txt", "w") as f:
        f.write(f"accuracy={acc:.4f}\nauc={auc:.4f}\n")

    # Log manual por si quieres además de autolog
    # mlflow.log_metric("accuracy_manual", acc)
    # mlflow.log_metric("auc_manual", auc)

    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
