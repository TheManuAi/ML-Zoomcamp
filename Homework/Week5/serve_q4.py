import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# --- Pydantic Model ---
# Defining the structure of the input data
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# --- Loading Model ---
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# --- Creating an App ---
app = FastAPI()

# --- Defining an Endpoint ---
@app.post("/predict")
def predict(client: Client):
    client_d = client.model_dump()

    probability = pipeline.predict_proba([client_d])[0][1]

    return {"conversion_probability": float(probability)}