from fastapi import APIRouter, HTTPException, Depends, Request
from .models import SingleQuery, BatchQuery, ClassificationResult
from ml.loader import get_models
from .auth import authenticate
import time

router = APIRouter()
start_time = time.time()

@router.get("/")
def read_root():    
    return {"message": "Welcome to the Intent Classification API. Visit /docs for API documentation."}

# Single Classification
@router.post("/api/classify", response_model=ClassificationResult)
def classify(query: SingleQuery, request: Request):
    try:
        vectorizer = request.app.state.vectorizer
        clf = request.app.state.classifier

        X = vectorizer.transform([query.text.strip()])   # convert to numeric
        intent = clf.predict(X)[0]
        confidence = float(max(clf.predict_proba(X)[0]))

        return {"intent": intent, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Classification
@router.post("/api/classify/batch")
def classify_batch(batch: BatchQuery, request: Request):
    try:
        vectorizer = request.app.state.vectorizer
        clf = request.app.state.classifier

        X = vectorizer.transform(batch.texts)
        intents = clf.predict(X)
        probs = clf.predict_proba(X)

        return [
            {"text": text, "intent": intent, "confidence": float(max(prob))}
            for text, intent, prob in zip(batch.texts, intents, probs)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Info
@router.get("/api/model/info")
def model_info(auth: bool = Depends(authenticate)):
    vectorizer, classifier = get_models()
    return {
        "model_type": str(type(classifier)),
        "vectorizer_type": str(type(vectorizer)),
        "classes": list(classifier.classes_),
        "status": "loaded"
    }


# Health Check
@router.get("/api/health")
def health_check(request: Request):
    uptime = round(time.time() - start_time, 2)

    vectorizer = getattr(request.app.state, "vectorizer", None)
    classifier = getattr(request.app.state, "classifier", None)

    # Check if model/vectorizer exist
    model_loaded = classifier is not None
    vectorizer_loaded = vectorizer is not None

    return {
        "status": "ok" if model_loaded and vectorizer_loaded else "degraded",
        "uptime_seconds": uptime,
        "model_loaded": model_loaded,
        "vectorizer_loaded": vectorizer_loaded
    }

