import joblib
import os

_vectorizer = None
_classifier = None

def get_models():
    global _vectorizer, _classifier
    if _vectorizer is None or _classifier is None:
        base_path = os.path.dirname(__file__)
        vec_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
        clf_path = os.path.join(base_path, "intent_classifier.pkl")
        
        _vectorizer = joblib.load(vec_path)
        _classifier = joblib.load(clf_path)
    
    return _vectorizer, _classifier
