from collections import OrderedDict
import os
import csv
import gzip
import pickle
import pandas as pd


def load_model(file):
    # Define the file path for the compressed model
    model_file_path = os.path.join(os.path.dirname(__file__), '..', 'classification_models', file)
    # Check if the file exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"The specified model file '{model_file_path}' does not exist.")
    # Load the compressed model file
    with gzip.open(model_file_path, 'rb') as f:
        loaded_model_bytes = f.read()
    # Unpickle the model bytes
    loaded_model = pickle.loads(loaded_model_bytes)

    return loaded_model


def load_preprocessing(imputer_cat_file, imputer_num_file, encoder_files, standardizer_file):
    # Define the file paths for preprocessing models
    imputer_cat_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', imputer_cat_file)
    imputer_num_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', imputer_num_file)
    encoder_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', encoder_files)
    standardizer_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', standardizer_file)

    # Load categorical imputer
    with gzip.open(imputer_cat_file_path, 'rb') as f:
        loaded_cat_imp = pickle.load(f)

    # Load numerical imputer
    with gzip.open(imputer_num_file_path, 'rb') as f:
        loaded_num_imp = pickle.load(f)

    # Load encoder
    with gzip.open(encoder_file_path, 'rb') as f:
        loaded_encoder = pickle.load(f)

    # Load standardizer
    with gzip.open(standardizer_file_path, 'rb') as f:
        loaded_standardizer = pickle.load(f)

    return loaded_cat_imp, loaded_num_imp, loaded_encoder, loaded_standardizer


def preprocess_data(df, column_names):
    # Load preprocessing objects
    trained_cat_imputer, trained_num_imputer, trained_onehot_encoder, trained_standardizer = load_preprocessing(
        'imputer_cat.pkl.gz', 'imputer_num.pkl.gz', 'encoder.pkl.gz', 'standardizer.pkl.gz')

    existing_columns = list(df.keys())
    # Find the missing columns
    missing_columns = [col for col in column_names if col not in existing_columns]

    # Add the missing columns to df and assign them a value of None
    for col in missing_columns:
        df[col] = None

    # Order the columns of df as per column_names
    df = OrderedDict((col, df[col]) for col in column_names)

    # Convert to DataFrame
    df = pd.DataFrame(df)

    # For categorical imputer (`trained_cat_imputer`)
    categorical_cols = trained_cat_imputer.get_feature_names_out()

    numerical_cols = [col for col in column_names if col not in categorical_cols]
    # Apply imputation for numerical features
    df[numerical_cols] = trained_num_imputer.transform(df[numerical_cols])

    # Apply imputation for categorical features
    df[categorical_cols] = trained_cat_imputer.transform(df[categorical_cols])

    # print(f"df: {df.head(20)}")
    # Apply one-hot encoding
    df_encoded = trained_onehot_encoder.transform(df[categorical_cols])
    df_encoded_df = pd.DataFrame(df_encoded, columns=trained_onehot_encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, df_encoded_df], axis=1)

    df = trained_standardizer.transform(df)

    return df


def predict_churn(df, model, column_names):
    # Preprocess the data
    df_processed = preprocess_data(df, column_names)

    # Make predictions
    predictions = model.predict(df_processed)

    return predictions


# # Load the trained model
# xgb_classifier = load_model('xgboost_classifier.pkl.gz')
#
# # Load column names
# column_names_path = os.path.join(os.path.dirname(__file__), '..', 'feature_names', 'original_feature_names.csv')
# column_names = pd.read_csv(column_names_path)['Feature Names'].tolist()
# print(f'Column names: {column_names}')
#
# # Load the new clients data
# new_clients_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'NewClients.csv')
# print("File path:", new_clients_file_path)  # Add this line to print the file path
# new_clients_df = pd.read_csv(new_clients_file_path)
#
# # Make predictions
# predictions = predict_churn(new_clients_df, xgb_classifier, column_names)
#
# print("Predictions for new clients:")
# print(predictions)
