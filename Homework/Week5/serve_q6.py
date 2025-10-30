import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI()

@app.post("/predict")
def predict(client: Client):
    client_d = client.model_dump()

    probability = pipeline.predict_proba([client_d])[0][1]
    return {"conversion_probability": float(probability)}