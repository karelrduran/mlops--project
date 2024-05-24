****[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.12.2-blue)](https://mlflow.org/)

<h1 align="center"> MLops Project: Churn Classification Model Deployment </h1>
<p align="center">
    <img src="assets/images/banner.png">
</p>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](##overview)
- [Overview](#overview)
- [Pre-requisites](##pre-requisites)
- [Dependencies](#dependencies)
- [Project structure](#project-structure)
- [Usage](#usage)
- [Personal situation](#personal-situation)

## Overview
You are a young start-up of machine learning engineers hired by a financial institution that operates in different countries around the world. While they have an amazing data science team, that has built a churn classification model, they are struggling to ensure that their work is deployed in a efficient and scalable way.
They have hired your team to make the work from their DS team *tracktable*, and *scalable* in a automated way. Your manager, has given you access to the repo from the team and your job is to make it production ready!

This project puts together the work that has been done on previous projects at Becode:

## Pre-requisites
This project require **MLflow ~= 2.12.2**, **Docker ~= 26.1.2** and **Terraform ~= 2.25.0** to be installed and running.

## Dependencies
    pandas~=2.2.1
    matplotlib~=3.8.2
    scikit-learn~=1.4.1.post1
    xgboost~=2.0.3
    imblearn~=0.0
    mlflow~=2.12.2
    optuna~=3.6.1

## Project structure
```
├── ml/
│   ├──classification_models/
│   │   └── xgboost_classifier.pkl.gz
│   ├── data/
│   │   ├── BankChurners.csv
│   │   └── NewClients.csv
│   ├──feature_names/
│   │   ├── encoded_feature_names.csv
│   │   └── original_feature_names.csv
│   ├── preprocess_models/
│   └── src/
│       └── churn_prediction.py
│       └── classifiers.py
│       └── preprocessing.py
├── mlflow-optuna.py
├── start.sh
├── README.md
├── main.tf
├── terraform.tfvars
└── variables.tf
```
## Usage

### 1. Set Up and Run MLflow

1. Clone the repository:

    ```bash
    git clone git@github.com:karelrduran/mlops--project.git
    cd mlops--project
    ```

2. Set up and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3. Install the Python dependencies:

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. Start MLflow to monitor your experiments:

    ```bash
    mlflow ui
    ```

### 2. Train and Register the Model

1. Run the training script to register the model in MLflow:
  
   Perform hyperparameter tuning:

    ```bash
    python mlflow-optuna.py
    ```

3. Package the model with MLflow:

    ```bash
    mlflow models serve -m models:/my_model/1
    ```

### 3. Deploy the Application with Docker and Terraform

1. Build the Docker image and deploy the container:

    ```bash
    terraform init
    terraform apply
    ```

2. Verify that the Docker container is running:

    ```bash
    docker ps
    ```

3. Access the service on the specified port (default is 5001):

    ```bash
    http://localhost:5001
    ```

## Personal situation
While doing this project I was part of the ARAI6 group of the <a href="https://becode.org/all-trainings/pedagogical-framework-ai-data-science/">AI Bootcamp</a> training organized by <a href="https://becode.org/">BeCode</a> in Ghent. The main objective of this project is to provide participants with an opportunity to understand the concepts of object-oriented programming (OOP) in Python. As well as getting information from an API, scraping a website that does not provide an API and saving the output for later use.


<img src="https://avatars.githubusercontent.com/u/106887418?s=400&u=82192b481d8f03c3eaad34ca2bd67889fce6a0c2&v=4" width=115><br><sub><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="Miniatura" width=20><a href="https://www.linkedin.com/in/karel-rodriguez-duran/">Karel Rodríguez Durán</a></sub>****

