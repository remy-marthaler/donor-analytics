# 🎯 Donor Analytics

**Donor Analytics** is a modular Streamlit application for analyzing fundraising and donation behaviour.  
It combines **Donor Segmentation**, **Churn Prediction**, and **Lifetime Value (LTV)** estimation into one intuitive interface.  
The goal is to help charities identify high-value donors, detect churn risks, and prioritise outreach efficiently.

![donor-analytics.png](donor-analytics.png)

---
## The Challenge Charities Face

Most charities struggle not because of a lack of generosity, but because they lack the ability to understand their own fundraising data. Donations come from many different people at irregular times and in different amounts, making it difficult to recognise patterns or spot risks. As a result, organisations often do not know which supporters are highly engaged, which ones are about to stop donating, or how much future revenue they can realistically expect. This uncertainty makes planning harder and can lead to preventable revenue losses, especially for small charities with limited resources.


---

## 📦 Features

### 🔍 Donor Segmentation
- RFM feature engineering  
- K-Means clustering  
- PCA cluster visualisation  
- Target list for outreach  

### 💰 Donor Lifetime Value (LTV)
- Heuristic and ML-based LTV  
- Feature importance  
- Revenue-at-risk estimation  
- Interactive What-If Simulator  

### 🔮 Churn Prediction
- Recency-based churn labelling  
- Logistic regression model  
- Risk map  
- High-risk donor list  

---

## 🧱 Architecture Overview

The app consists of three analytics modules:

- **Segmentation** → `segmentation_page.py`
- **Churn Prediction** → `churn_page.py`
- **LTV Modelling** → `ltv_page.py`

All modules load their data from the **API layer**:
 - src/data_access/api_client.py → Real API (FundraisingBox)
 - src/data_access/mock_api_client.py → Mock API (local CSV)


---

## 🔌 Data Sources (Mock API vs Real API)

The system supports two data modes:


### 1) Mock API (Default for Local Development)

When no FundraisingBox API key is found, the application automatically uses **an anonymised CSV file** via: `MockApiClient → transactions_faked.csv`

Why this exists:
- Confidential data stays secure
- No need for API calls during development  
- Consistent, stable dataset  
- Faster iteration and debugging  
- Identical interface to the real API

The Mock API:
- Loads CSV data  
- Cleans German-formatted amounts (`1.234,56`)  
- Maps all fields to the internal schema  
- Acts as a drop-in replacement for live production data  


### 2) Real API (FundraisingBox)

To use live data, a `.env` file is required with the following content: `FBOX_API_KEY=your-real-api-key`


When this variable is present, the app switches to the **FundraisingBox API client**:

- Uses the live FundraisingBox API endpoints like `/donations.json` with pagination  
- Retries on rate limits  
- Cleans and maps fields  
- Caches responses for **12 hours** (to avoid expensive repeated calls)

This means:
- Local dev → Fake CSV (fast, safe)
- Production → Real fundraising data

---

## ⚙️ Installation & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch application
```bash
python -m streamlit run main.py
```

### 3. (Optional) Enable live data
```bash
export FBOX_API_KEY="your-key"
```

If the key is not set:
 - ✔ App still works
 - ✔ Uses anonymised mock data

## 📁 Project Structure
```

pages/
 │   ├─ segmentation_page.py
 │   ├─ churn_page.py
 │   ├─ ltv_page.py
 │
src/
 ├─ core/
 │   ├─ state.py              → API client selection (mock vs real)
 │   ├─ layout.py             → Shared UI elements
 │
 ├─ data_access/
 │   ├─ api_client.py         → Real FundraisingBox API
 │   ├─ mock_api_client.py    → CSV-based offline API


```

## 👥 Contribution Matrix

See [Contribution Matrix](CONTRIBUTIONS.md) for a full breakdown of team contributions.

GitHub repository: https://github.com/remy-marthaler/donor-analytics

## 📝 License

Internal student project.
Not intended for public distribution.
