import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import pickle
import os
import base64

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model using optimal k (elbow method),
    saves it, and returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    sse = []
    for k in range(1, 11):  # reduced range for stability
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # Find optimal k

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow if kl.elbow else 3  # fallback safe default

    print(f"Using optimal k = {optimal_k}")

    # Train final model
    final_model = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
    final_model.fit(df)

    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(final_model, f)

    return sse



def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    k_range = range(1, len(sse) + 1)
    kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
    
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        # if not numeric, still return a JSON-friendly version
        return pred.item() if hasattr(pred, "item") else pred
    

def evaluate_model(data_b64: str, filename: str):
    """
    Loads model, computes silhouette score,
    saves cluster distribution and simple scatter plot.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(model_path, "rb"))

    labels = model.predict(df)

    score = silhouette_score(df, labels)
    print(f"Silhouette Score: {score}")

    # Save cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = pd.DataFrame({
        "Cluster": unique,
        "Count": counts
    })

    output_dir = os.path.join(os.path.dirname(__file__), "../model")
    distribution.to_csv(os.path.join(output_dir, "cluster_distribution.csv"), index=False)

    # Simple scatter plot (first two features)
    plt.figure()
    plt.scatter(df[:, 0], df[:, 1], c=labels)
    plt.title("Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(os.path.join(output_dir, "cluster_plot.png"))
    plt.close()

    return float(score)

