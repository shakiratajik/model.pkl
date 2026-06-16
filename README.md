# Customer Churn Prediction Project

A simple Machine Learning project with a working website (GUI) built using Streamlit.

## What is inside this folder

| File | What it does |
|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Your dataset |
| `backend.py` | Trains the ML model and saves it as `model.pkl` (run this FIRST) |
| `app.py` | The website (frontend + GUI) — loads `model.pkl` and lets users predict churn |
| `requirements.txt` | List of Python packages needed |
| `model.pkl` | Created automatically after you run `backend.py` (the saved trained model) |

## Why two files instead of one?

In your original code, everything (loading data, training, testing) was in one script.
That is fine for testing, but not for a real project with a website.

So we split it into two parts:

1. **Backend (`backend.py`)** — does all the heavy lifting: cleans data, trains the
   model, checks accuracy, and saves the trained model to a file called `model.pkl`.
   You only need to run this once (or whenever you change the dataset).

2. **Frontend / GUI (`app.py`)** — this is the actual website. It does NOT train
   anything. It just opens `model.pkl`, takes the customer details you type in
   through dropdowns and sliders, and shows you the prediction instantly.

This is how real-world ML apps are built: train once, then reuse the saved model.

## Step-by-step: how to run this on your own computer

### Step 1: Install Python
Make sure Python 3.9 or newer is installed. Check with:
```
python --version
```

### Step 2: Open a terminal in this project folder
Put all 4 files (`backend.py`, `app.py`, `requirements.txt`, and the CSV file)
in the SAME folder. Then open a terminal/command prompt inside that folder.

### Step 3: Install the required packages
```
pip install -r requirements.txt
```

### Step 4: Run the backend (this trains the model)
```
python backend.py
```
You should see accuracy numbers printed, and a new file called `model.pkl`
will appear in the folder. This is your trained "brain".

### Step 5: Run the frontend (this opens your website)
```
streamlit run app.py
```
Your browser will automatically open a page that looks like a small website.
This is your GUI.

### Step 6: Use the website
- Fill in the customer details using the dropdown menus and sliders.
- Click the "Predict Churn" button.
- You will instantly see whether the customer is likely to leave (Churn)
  or stay, along with a percentage probability.

## How to put this online (so others can use it without installing anything)

The easiest free option is **Streamlit Community Cloud**:

1. Create a free GitHub account (if you don't have one) and create a new
   repository, e.g. `churn-prediction-app`.
2. Upload these 4 files to that GitHub repository:
   - `app.py`
   - `backend.py`
   - `requirements.txt`
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Also upload the `model.pkl` file you created in Step 4 above (so the
   website doesn't need to retrain itself online).
4. Go to https://share.streamlit.io and sign in with GitHub.
5. Click "New app", select your repository, and set the main file as `app.py`.
6. Click "Deploy". After a minute, you will get a public link like
   `https://your-app-name.streamlit.app` that you can share with your
   teacher, classmates, or in your project report.

## Notes for your project report

- **Backend**: `backend.py` — handles data cleaning, encoding, scaling,
  model training, and evaluation (Accuracy, Precision, Recall, F1 Score,
  Confusion Matrix).
- **Frontend / GUI**: `app.py` — a Streamlit web interface where a user
  enters customer information and receives a real-time churn prediction.
- **Model used in final app**: Random Forest Classifier (chosen because it
  performed well among the models tested: Logistic Regression, Decision
  Tree, Random Forest, KNN, Naive Bayes).
- You can mention in your report that clustering (K-Means and Hierarchical)
  was also explored separately for customer segmentation, in addition to
  the churn classification task.
