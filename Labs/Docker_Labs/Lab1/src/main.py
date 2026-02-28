# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import numpy as np

if __name__ == '__main__':
    # Load the Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=wine.target_names)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    header = "            " + "  ".join(f"{name:>9}" for name in wine.target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>9}" for val in row)
        print(f"{wine.target_names[i]:>12}  {row_str}")

    # Save metrics to JSON
    report_dict = classification_report(y_test, y_pred, target_names=wine.target_names, output_dict=True)
    metrics = {
        "dataset": "wine",
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "test_size": 0.2,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to metrics.json")

    # Save the model to a file
    joblib.dump(model, 'wine_model.pkl')
    print("Model saved as wine_model.pkl")
