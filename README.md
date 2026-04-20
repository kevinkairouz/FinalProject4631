# Social Media & Productivity Score Predictor — Web App

A full-stack web application that predicts a user's productivity score based on their daily habits and social media usage. Users can choose between two ML models — a Gradient Boosting Regressor and a PyTorch Neural Network — and receive an instant prediction through an interactive frontend.

> **Note:** For the ML analysis and EDA only, see the `main` branch.

---

## Features
- **Dual Model Predictions:** Choose between Gradient Boosting or Neural Network via dedicated buttons
- **Flask REST API:** Two separate POST endpoints, one per model
- **Prediction Logging:** Every prediction is stored in a MySQL database
- **Interactive Frontend:** HTML/CSS/JS interface with animated UI

---

## Project Structure

```
├── model/
│   ├── model.py              # Gradient Boosting model + predict function
│   └── neuralNetwork.py      # PyTorch Neural Network + predict function
├── data/
│   └── social.csv            # Kaggle dataset: Social Media & Productivity
├── views/
│   ├── index.html            # Frontend UI
│   ├── index.css             # Frontend styles
│   └── index.js              # Frontend logic (fetch API calls)
├── app.py                    # Flask application entry point
└── requirements.txt          # Required libraries
```

---

## Flask API Architecture

**Base URL:** `http://127.0.0.1:5000`

| Route | Method | Description |
|---|---|---|
| `/` | GET | Welcome message |
| `/predictGB` | POST | Returns productivity score from Gradient Boosting model |
| `/predictNN` | POST | Returns productivity score from Neural Network model |

**Request Format:**
```json
{
  "age": 22,
  "ScreenTime": 6.5,
  "SocialHours": 3.0,
  "StudyHours": 4.0,
  "SleepHours": 7.0,
  "Noti": 45
}
```

**Response Format:**
```json
{
  "productivity_score": 63.4
}
```

---

## Model Evaluation

| Model | R² Score | MAE |
|---|---|---|
| Neural Network | ~88.4% | ~7.2 |
| Gradient Boosting | ~88.2% | ~8.0 |

---

## Installation & Setup

**Prerequisites:**
- Python 3.11+
- MySQL (local server running)
- pip

**Install dependencies:**
```bash
pip install flask flask-cors pandas numpy scikit-learn matplotlib torch torchmetrics mysql-connector-python
```

---

## MySQL Setup

Create the database and table before running the app:

```sql
CREATE DATABASE finalProj;

USE finalProj;

CREATE TABLE user_predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  age INT,
  daily_screen_time FLOAT,
  social_media_hours FLOAT,
  study_hours FLOAT,
  sleep_hours FLOAT,
  notifications_per_day INT,
  productivity_score FLOAT
);
```

Update the credentials in `model/model.py` to match your local MySQL setup:
```python
db = sql.connect(host="localhost", user="root", password="your_password", database="finalProj")
```

---

## How to Run

```bash
python app.py
```

Then open `views/index.html` in your browser and make sure the Flask server is running at `http://127.0.0.1:5000`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML (Classical) | scikit-learn (GradientBoostingRegressor) |
| ML (Deep Learning) | PyTorch, torchmetrics |
| Data | pandas, NumPy |
| Backend | Flask, Flask-CORS |
| Database | MySQL |
| Frontend | HTML, CSS, JavaScript |
