Airflow Lab 1: Enhanced Clustering Pipeline
Overview

This lab implements a machine learning workflow using Apache Airflow. The pipeline performs data loading, preprocessing, optimal cluster selection using the elbow method, model training, and evaluation.

I extended the original lab by correcting the clustering logic and adding a model evaluation stage to improve interpretability and robustness.

What the Pipeline Does
1. The DAG Airflow_Lab1 runs the following tasks:
2. Load data from a CSV file
3. Preprocess and scale selected features
4. Compute the optimal number of clusters using the elbow method
5. Retrain KMeans using the selected number of clusters
6. Save the trained model
7. Evaluate clustering performance using Silhouette Score
8. Generate clustering artifacts

Task order:
1. load_data
2. data_preprocessing
3. build_save_model
4. load_model
5. evaluate_model

Improvements Made

1. This version differs from the original lab in the following ways:
2. The optimal number of clusters is selected dynamically using KneeLocator
3. The model is retrained using the detected optimal K instead of the maximum cluster value
4. A new evaluation task was added to compute Silhouette Score
5. The pipeline now generates a cluster distribution CSV file
6. A cluster visualization plot is saved as a PNG file
7. The SSE range handling was made dynamic to prevent range mismatches

These changes improve correctness and provide basic model evaluation and artifact generation.

Project Structure

Lab_1
dags
airflow.py
data
src
lab.py
docker-compose.yaml
setup.sh
README.md

Requirements

Docker Desktop installed
At least 4GB RAM allocated to Docker

How to Run
1. From the project root directory:
    docker compose up
2. Wait until the webserver is ready.
    Open:
    http://localhost:8080
3. Login with:
    Username airflow
    Password airflow
4. Enable the DAG Airflow_Lab1 and trigger it manually.

All tasks should complete successfully.

Output

After execution, the following files are generated inside the dags/model directory:

model.sav
cluster_distribution.csv
cluster_plot.png

Summary

This lab demonstrates how to orchestrate a machine learning pipeline using Airflow and Docker. The added evaluation stage and corrected cluster selection improve both the reliability and interpretability of the workflow.