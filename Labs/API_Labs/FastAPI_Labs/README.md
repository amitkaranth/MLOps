
# FastAPI Wine Classification Lab

This project serves a machine learning classification model as a REST API using FastAPI.  
It is a modified version of the original FastAPI lab and uses the Wine dataset instead of Iris.

---

## What Changed

- Switched dataset from Iris to Wine  
- Trained a new classification model on the Wine dataset  
- Updated input schema to support 13 wine features  
- Added two new API endpoints:
  - `GET /model-info`
  - `POST /predict-batch`
- Saved and used a new model artifact (`wine_model.pkl`)

---

## Project Structure

```text
Fastapi_Lab/
├── assets/
│   ├── batch_predictions.png
│   ├── front_page.png
│   ├── model-info.png
│   ├── prediction_1.png
│   └── prediction_2.png
├── model/
│   └── wine_model.pkl
├── src/
│   ├── data.py
│   ├── train.py
│   ├── predict.py
│   └── main.py
├── requirements.txt
└── README.md
```

---

## Setup and Run

Create and activate a virtual environment:

```bash
python -m venv fastapi_lab_env
source fastapi_lab_env/bin/activate   # macOS/Linux
fastapi_lab_env\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
cd src
python3 train.py
```

Run the API:

```bash
uvicorn main:app --reload
```

Open the API docs at:

```
http://127.0.0.1:8000/docs
```

---

## Added Endpoints

* `GET /model-info`
  Returns basic information about the trained model.

* `POST /predict-batch`
  Accepts multiple samples and returns predictions for each.

---

## Example Predict Input Request

```json
{
  "alcohol": 14,
  "malic_acid": 1.5,
  "ash": 3.2,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127,
  "total_phenols": 2.5,
  "flavanoids": 3.2,
  "nonflavanoid_phenols": 0.2,
  "proanthocyanins": 2.3,
  "color_intensity": 5.2,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.5,
  "proline": 1047
}
```
The above example gives an prediction response: 0
The screenshot of the input/ output is available in the assets/ folder

```json
{
  "alcohol": 13.44,
  "malic_acid": 5.22,
  "ash": 2.56,
  "alcalinity_of_ash": 20.6,
  "magnesium": 92,
  "total_phenols": 1.8,
  "flavanoids": 0.62,
  "nonflavanoid_phenols": 0.52,
  "proanthocyanins": 1.06,
  "color_intensity": 7.1,
  "hue": 0.7,
  "od280_od315_of_diluted_wines": 1.75,
  "proline": 747
}
```
The above example gives an prediction response: 2
The screenshot of the input/ output is available in the assets/ folder

## Example Batch Request

```json
{
  "samples": [
    [14, 1.5, 3.2, 15.6, 127, 2.5, 3.2, 0.2, 2.3, 5.2, 1.05, 3.5, 1047],
    [13.44, 5.22, 2.56, 20.6, 92, 1.8, 0.62, 0.52, 1.06, 7.1, 0.7, 1.75, 747]
  ]
}
```

```json
{
  "predictions": [0, 2]
}
```

---

Screenshots of the API and responses are included in the `assets/` folder.

```
```
