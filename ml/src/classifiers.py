import os
import pandas as pd
from .preprocessing import DataPreprocessor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import gzip


def xgboost_classifier(X_train, X_test, y_train, y_test, feature_names_original, params):
    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)  # Train XGBoost on preprocessed data
    # print("XGBoost Classifier:")
    # print(classification_report(y_test, xgb.predict(X_test)))
    feature_importance_dict = dict(zip(feature_names_original, xgb.feature_importances_))
    return xgb, feature_importance_dict


def roc_c(X_test, y_test, clf, clf_name):
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"{clf_name} ROC AUC score: {roc_auc}")


def confusion_m(X_test, y_test, clf, clf_name):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{clf_name} Confusion matrix:")
    print(cm)
    print()


def cross_val(X_train, y_train, clf, clf_name):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{clf_name} Cross-validation scores:", scores)
    print(f"{clf_name} Mean cross-validation score:", scores.mean())
    print()


def plot_feature_importance(feature_importance):
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)
    num_features = len(features)

    # Calculate the appropriate figure size based on the number of features
    figsize_height = min(0.5 * num_features, 10)  # Maximum height of 10 inches
    plt.figure(figsize=(10, figsize_height))

    plt.barh(range(num_features), importance, align='center')
    plt.yticks(range(num_features), features)  # Set the feature names as y-axis ticks
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importance')

    # Adjust the layout to avoid clipping feature names
    plt.tight_layout()

    # Save the plot as an image file
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_file_path = os.path.join(output_folder, 'feature_importance_plot.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Feature importance plot saved to {plot_file_path}")


# Function to export each model to a separate file
def export_model(model, model_name, export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    filename = os.path.join(export_folder, f"{model_name}.pkl.gz")
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)
    # print(f"Model {model_name} exported to {filename}")
