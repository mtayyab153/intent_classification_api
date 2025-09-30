from fastapi import FastAPI
from . import endpoints
from ml.loader import get_models

app = FastAPI(
    title="Intent Classification API",
    description="FastAPI backend for intent classifier",
    version="1.0.0"
)


@app.on_event("startup")
def load_model():
    app.state.vectorizer, app.state.classifier = get_models()


app.include_router(endpoints.router)
