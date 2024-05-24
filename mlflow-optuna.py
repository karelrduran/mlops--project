import os
import pandas as pd
import optuna
from mlflow.models import infer_signature
from xgboost import XGBClassifier

from ml.src.classifiers import export_model
from ml.src.preprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score

import mlflow


# Load data

current_directory = os.path.dirname(__file__)
csv_file_path = os.path.join(current_directory, "ml", "data", "BankChurners.csv")
churners_df = pd.read_csv(csv_file_path, sep=',')

# Instantiate the class with the dataframe name
preprocessor = DataPreprocessor(df_name=churners_df)

# Use the preprocess method to get preprocessed data
X_train, X_test, y_train, y_test = preprocessor.preprocess()

optuna.logging.set_verbosity(optuna.logging.ERROR)

# Set the tracking server usi for logging
mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLflow Experiment
mlflow.set_experiment("MLOps Churn Predict")


def objective(trial):
    params = {
        "booster": trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart']),
        "learning_rate": trial.suggest_float("learning_rate", 0, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 30),
        "objective": trial.suggest_categorical("objective", ['reg:logistic', 'binary:logistic', 'binary:logitraw']),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1000),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1000),
    }

    with mlflow.start_run(nested=True) as run:
        xgb = XGBClassifier(**params)
        xgb.fit(X_train, y_train)

        # # Save the model in JSON format
        # xgb.save_model("model.json")
        #
        # # Log the saved model
        # mlflow.log_artifact("model.json")

        ypred = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, ypred)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

    return accuracy


# Start an MLflow run
mlflow.autolog()
with mlflow.start_run() as run:
    study = optuna.create_study(direction="maximize")

    # Log the loss metric
    # mlflow.log_metric("accuracy", accuracy)
    study.optimize(objective, n_trials=10)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic Classification model")

    best_xgboost_model = XGBClassifier(**study.best_params)
    best_xgboost_model.fit(X_train, y_train)

    # Infer the model signature
    signature = infer_signature(X_train, best_xgboost_model.predict(X_train))

    ypred = best_xgboost_model.predict(X_test)

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=XGBClassifier,
        artifact_path="churn_predict",
        signature=signature,
        input_example=X_train,
        registered_model_name="XGBoostClassifier_Model",
    )

    # Export the model to a file
    export_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml', 'classification_models'))

    export_model(best_xgboost_model, "xgboost_classifier", export_folder)
