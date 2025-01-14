from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}


from joblib import load

# Load the trained model
model = load('models/best_gb_model.pkl')


from pydantic import BaseModel

# Define the input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median Income must be greater than 0")
    HouseAge: float = Field(ge=0, description="House Age must be 0 or higher")
    AveRooms: float = Field(gt=0, description="Average Rooms must be greater than 0")
    AveBedrms: float = Field(gt=0, description="Average Bedrooms must be greater than 0")
    Population: float = Field(gt=0, description="Population must be greater than 0")
    AveOccup: float = Field(gt=0, description="Average Occupancy must be greater than 0")
    Latitude: float = Field(ge=-90, le=90, description="Latitude must be between -90 and 90")
    Longitude: float = Field(ge=-180, le=180, description="Longitude must be between -180 and 180")

# Prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to a list of features
    input_data = [[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]]

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result
    return {"predicted_price": prediction[0]}


