import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str | None
    utm_medium: str | None
    utm_campaign: str | None
    utm_adcontent: str | None
    utm_keyword: str | None
    device_category: str | None
    device_os: str | None
    device_brand: str | None
    device_model: str | None
    device_screen_resolution: str | None
    device_browser: str | None
    geo_country: str | None
    geo_city: str | None


class Prediction(BaseModel):
    session_id: str
    client_id: str
    pred: int


app = FastAPI()
with open('model/sber_auto_pipe.pkl', 'rb') as fh:
    model = dill.load(fh)


@app.get('/status')
def status():
    return 'OK'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/prediction', response_model=Prediction)
def prediction(form: Form):
    df = pd.DataFrame([form.dict()])
    y = model['model'].predict(df)
    return {
        'session_id': form.session_id,
        'client_id': form.client_id,
        'pred': y[0]
    }
