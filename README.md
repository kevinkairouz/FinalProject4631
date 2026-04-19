# Social Media & Productivity Score Predictor

A full-stack machine learning application that predicts a user's productivity score based on their daily habits and social media usage. Features a dual-model comparison between a Gradient Boosting Regressor and a PyTorch Neural Network, with a Flask REST API and interactive web frontend.

---

## Features
- **Dual ML Models:** Gradient Boosting Regressor (scikit-learn) and a Fully Connected Neural Network (PyTorch)
- **Model Comparison:** Side-by-side evaluation using R² Score and Mean Absolute Error (MAE)
- **Hyperparameter Tuning:** GridSearchCV used to identify optimal parameters for Gradient Boosting
- **Exploratory Data Analysis:** Scatter plots, histograms, and subplots visualizing key feature relationships
- **Real-time Predictions:** Web interface for users to input their habits and receive an instant productivity score
- **Prediction Logging:** MySQL database stores every prediction made via the API

---

## Project Structure

```
├── model/
│   ├── model.py              # Gradient Boosting model + Flask API predict logic + MySQL logging
│   ├── neuralNetwork.py      # PyTorch Neural Network training and evaluation
│   ├── analysis.py           # EDA visualizations and model comparison charts
│   ├── histogram.py          # Age distribution histogram
│   └── console.py            # Entry point for running all analysis scripts
├── data/
│   └── social.csv            # Kaggle dataset: Social Media & Productivity
├── images_analysis/
│   └── nn_image.png          # Neural Network architecture diagram
├── app.py                    # Flask application entry point
├── index.html                # Frontend UI
├── index.css                 # Frontend styles
├── index.js                  # Frontend logic (fetch API calls)
└── requirements.txt          # Required libraries
```

---

## Machine Learning Components

### Models Evaluated
- **GradientBoostingRegressor** — Sequential ensemble method; each tree corrects its predecessor
- **Neural Network (PyTorch)** — Fully connected feed-forward network with ReLU activations

### Gradient Boosting — Hyperparameter Tuning
Tuned via `GridSearchCV`:
- `n_estimators`: 150
- `learning_rate`: 0.1

### Neural Network Architecture
```
Input Layer:   6 nodes  (one per feature)
Hidden Layer 1: 4 nodes + ReLU
Hidden Layer 2: 4 nodes + ReLU
Hidden Layer 3: 4 nodes + ReLU
Output Layer:  1 node   (productivity score)

Optimizer: SGD (lr=0.001)
Loss:      MSELoss
Epochs:    100
Batch Size: 32
```

### Model Evaluation

| Model | R² Score | MAE |
|---|---|---|
| Neural Network | ~88.4% | ~7.2 |
| Gradient Boosting | ~88.2% | ~8.0 |

---

## Data & Preprocessing

**Dataset:** [Social Media & Productivity — Kaggle](https://www.kaggle.com)

**Features used (X):**
- `age`
- `daily_screen_time`
- `social_media_hours`
- `study_hours`
- `sleep_hours`
- `notifications_per_day`

**Target (Y):** `productivity_score`

**Preprocessing steps:**
- Dropped null rows with `dropna()`
- Removed extraneous columns: `addiction_level`, `focus_score`
- Cast `age` and `notifications_per_day` to integer with `pd.to_numeric`
- Applied `StandardScaler` for Neural Network input
- Data placed into `TensorDataset` → `DataLoader` (batch_size=32) for PyTorch training
- 70/30 train-test split (`random_state=42`)

---

## Flask API Architecture

**Base URL:** `http://127.0.0.1:5000`

| Route | Method | Description |
|---|---|---|
| `/` | GET | Welcome message |
| `/predict` | POST | Returns predicted productivity score |
| `/test` | GET | Returns test value |

**Request Format (`/predict`):**
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

Every prediction is automatically logged to a MySQL database (`finalProj.user_predictions`).

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

**How to Run:**
```bash
python console.py
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

Update the credentials in `model.py` to match your local MySQL setup:
```python
db = sql.connect(host="localhost", user="root", password="your_password", database="finalProj")
```

---

## Visualizations

### Neural Network Architecture
![Neural Network Architecture](images_analysis/nn_image.png)

### Exploratory Data Analysis
![Data Analysis Subplots](images_analysis/Data_Used_for_paper_visual.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML (Classical) | scikit-learn (GradientBoostingRegressor, GridSearchCV) |
| ML (Deep Learning) | PyTorch, torchmetrics |
| Data | pandas, NumPy |
| Visualization | matplotlib |
| Backend | Flask, Flask-CORS |
| Database | MySQL |
| Frontend | HTML, CSS, JavaScript |
