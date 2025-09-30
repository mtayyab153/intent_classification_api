# 📌 Intent Classifier API  

A FastAPI-based service for classifying text queries into intents using a trained machine learning model (**TF-IDF + Logistic Regression**).  

---

## 🚀 Features  
- Single & batch text classification  
- Model caching on startup for fast inference  
- Basic authentication for model management endpoints  
- Docker support for containerized deployment  
- Pytest-based tests included 

---

## 🛠️ Setup Instructions  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/intent-classifier-app.git
cd intent-classifier-app
```

### 2. Create Virtual Environment  
```bash
python -m venv venv
# Activate venv (Windows)
venv\Scripts\activate
# Activate venv (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

## ▶️ Run Locally  

```bash
uvicorn api.main:app --reload --port 8000
```

Now open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.  

---

## 🧪 Running Tests  

```bash
pytest
```

---

## 🐳 Docker Deployment  

### 1. Build Image  
```bash
docker build -t intent-classifier .
```

### 2. Run Container  
```bash
docker run -d -p 8000:8000 intent-classifier
```

### 3. Access API  
Visit [http://localhost:8000/docs](http://localhost:8000/docs).  

---

## 🔑 Authentication  

For protected endpoints (like model info/management), use **Basic Auth** in Swagger (lock icon 🔒).  

- **Username:** `admin`  
- **Password:** `admin@786`  
(Stored in `auth.py`, can be updated as needed.)  

---

## 📂 Project Structure  

```
intent_classifier_app/
│── ml/
│   ├── intent_classifier.pkl          
│   ├── loader.py
│   ├── intent_classifier.ipynb          
│   ├── tfidf_vectorizer.pkl     
│── api/
│   ├── main.py          # FastAPI entrypoint
│   ├── auth.py          # Basic authentication
│   ├── endpoints.py     # API endpoints
│   ├── models.py        # Pydantic schemas
│   └── tests/           # Pytest test cases
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```