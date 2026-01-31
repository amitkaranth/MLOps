from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
import joblib

app = FastAPI()

model = joblib.load("../model/wine_model.pkl")

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class WineResponse(BaseModel):
    response: int


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(data: WineData):
    try:
        features = [[data.alcohol, data.malic_acid, data.ash,
                    data.alcalinity_of_ash, data.magnesium,
                    data.total_phenols, data.flavanoids,
                    data.nonflavanoid_phenols, data.proanthocyanins,
                    data.color_intensity, data.hue,
                    data.od280_od315_of_diluted_wines,
                    data.proline]]

        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from predict import WineBatchRequest

@app.post("/predict-batch")
def predict_batch(data: WineBatchRequest):
    predictions = model.predict(data.samples)
    return {
        "predictions": predictions.tolist()
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "dataset": "Wine",
        "num_features": 13,
        "num_classes": 3
    }




    
