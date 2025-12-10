import sys
import os

from src.core.state import get_api_client

# --- 1. SETUP: Fix imports so we can find the API ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# Import the existing API client provided by your team
from src.data_access.mock_api_client import MockApiClient

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Churn Prediction", page_icon="🔮")

def main():
    # --- REQUIREMENT 1: CLEARLY STATE THE PROBLEM ---
    st.markdown("""
    **The Problem:** Charities lose money when donors stop giving ("Churn").\n
    **The Solution:** We use Machine Learning to predict who is at risk so we can contact them.
    """)

    # --- REQUIREMENT 2: LOAD DATA VIA API ---
    @st.cache_data
    def get_data():
        api = get_api_client()
        # Get raw data
        df = api.get_donations()
        
        # CLEANING: Fix weird date formats and money strings (e.g. "50,00")
        df["date"] = pd.to_datetime(df["Getätigt am Datum"], dayfirst=True, errors='coerce')
        # Normalize Betrag ONLY if it is a string (from CSV)
        # NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025).
        # Specifically, the logic to handle European number formats (removing dots, replacing commas).
        # The authors reviewed and validated the final logic.
        if not pd.api.types.is_numeric_dtype(df["Betrag"]):
            amount_str = df["Betrag"].astype(str)
            amount_str = amount_str.str.replace(".", "", regex=False)
            amount_str = amount_str.str.replace(",", ".", regex=False)
            df["Betrag"] = pd.to_numeric(amount_str, errors="coerce")
        else:
            df["Betrag"] = pd.to_numeric(df["Betrag"], errors="coerce")
            
        return df

    try:
        raw_df = get_data()
    except Exception as e:
        st.error("Could not load data. Check API.")
        return

    # --- REQUIREMENT 4: USER INTERACTION (The Slider) ---
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Define Churn (Days without donation)", 
        min_value=90, max_value=730, value=365
    )

    # --- DATA PREPARATION (Simplifying the Logic) ---
    # We group data by Donor ID to get stats: Recency (Days since last), Frequency (Count), Monetary (Average)
    # We use the dataset's max date as "Today" so the math works
    today = raw_df["date"].max()
    
    donors = raw_df.groupby("Kontakt-ID").agg({
        "date": "max",          # Last donation date
        "Betrag": "mean",       # Average donation amount
        "Kontakt-ID": "count"   # Number of donations (Frequency)
    }).rename(columns={"Kontakt-ID": "frequency", "Betrag": "avg_amount"})
    
    # Calculate "Recency" (Days since last donation)
    donors["recency"] = (today - donors["date"]).dt.days
    
    # Define who has ALREADY churned (The "Truth" for training)
    donors["is_churned"] = (donors["recency"] > threshold).astype(int)

    # --- REQUIREMENT 5: MACHINE LEARNING ---
    # We teach the computer: "Look at Amount and Frequency. Does this usually lead to Churn?"
    # NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025).
    # Specifically, the setup of the Logistic Regression model and the extraction of probabilities using predict_proba.
    # The authors reviewed and validated the final logic.
    
    # 1. Prepare inputs (X) and target (y)
    X = donors[["avg_amount", "frequency"]]
    y = donors["is_churned"]
    
    # 2. Train the model (Logistic Regression is standard for Yes/No predictions)
    model = LogisticRegression()
    model.fit(X, y)
    
    # 3. Predict the probability of churning for EVERYONE
    donors["churn_prob"] = model.predict_proba(X)[:, 1]
    
    # --- REQUIREMENT 3: VISUALIZATION ---
    st.divider()
    
    # Show key metrics
    col1, col2 = st.columns(2)
    risk_count = len(donors[(donors["is_churned"] == 0) & (donors["churn_prob"] > 0.7)])
    
    col1.metric("Current Churn Rate", f"{(y.mean()*100):.1f}%")
    col2.metric("⚠️ Active Donors at Risk", risk_count, help="Active donors with >70% churn probability")

    # Visual 1: The "Risk Map"
    st.subheader("📊 Donor Risk Map")
    st.caption("Donors in the top-left (Low Frequency) are at higher risk.")
    
    fig = px.scatter(
        donors,
        x="frequency",
        y="avg_amount",
        color="churn_prob",
        title="Machine Learning Risk Prediction",
        labels={"churn_prob": "Risk Score (0-1)", "frequency": "Number of Donations"},
        color_continuous_scale="RdBu_r" # Red = High Risk
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ACTIONABLE LIST (Business Value) ---
    st.subheader("🚨 High Priority Contact List")
    st.caption("These donors are currently **Active** but the AI thinks they might leave soon.")
    
    # Filter: Active donors (not churned yet) with High Probability (> 70%)
    at_risk = donors[
        (donors["is_churned"] == 0) & 
        (donors["churn_prob"] > 0.70)
    ].sort_values("avg_amount", ascending=False)

    # NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025).
    # Specifically, the configuration of the 'ProgressColumn' for visualizing probabilities in the dataframe.
    # The authors reviewed and validated the final logic.
    
    st.dataframe(
        at_risk[["avg_amount", "frequency", "recency", "churn_prob"]],
        use_container_width=True,
        column_config={
            "churn_prob": st.column_config.ProgressColumn("Risk Probability", format="%.2f"),
            "avg_amount": st.column_config.NumberColumn("Avg Donation", format="CHF %.2f"),
        }
    )
    
    # --- REQUIREMENT 6: DOCUMENTATION ---
    # (Comments are included above in the code to satisfy this requirement)

if __name__ == "__main__":
    main()
