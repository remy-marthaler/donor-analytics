import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.core.state import get_api_client

# ----- Layout -----
st.caption("Segments donors into clusters to prioritize outreach.")

api = get_api_client()

# Step A: loading data
# adjust calls to the API
donations = api.get_donations()  # expects list from dicts

df = pd.DataFrame(donations)
# map CSV-column to intern standard names
column_mapping = {
    "Kontakt-ID": "donor_id",
    "Getätigt am Datum": "donation_date",
    "Betrag": "amount",
    "Vorname": "first_name",
    "Nachname": "last_name",
}

# test, if these original columns exist
missing_orig = [c for c in column_mapping.keys() if c not in df.columns]
if missing_orig:
    st.error(f"Es fehlen folgende Spalten: {missing_orig}")
    st.stop()

# rename columns
df = df.rename(columns=column_mapping)

# --- Cleaning ---
# Convert raw date strings into datetime objects, coercing invalid formats
df["donation_date"] = pd.to_datetime(
    df["donation_date"], errors="coerce")

# normalise amounts: only if not numeric (z. B. CSV)
# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
if not pd.api.types.is_numeric_dtype(df["amount"]):
    amount_str = df["amount"].astype(str)
    amount_str = amount_str.str.replace(".", "", regex=False)
    amount_str = amount_str.str.replace(",", ".", regex=False)
    df["amount"] = pd.to_numeric(amount_str, errors="coerce")
else:
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# delete invalid or negative values
df = df.dropna(subset=["donor_id", "donation_date", "amount"])
df = df[df["amount"] > 0]

if df.empty:
    st.warning("No valid donations found after cleaning.")
    st.stop()

# Step B: RFM features per donor
# Compute RFM features per donor (recency, frequency, monetary value)
# Reference = last donor timing in dataset
ref_date = df["donation_date"].max()

# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
rfm = (
    df.groupby("donor_id")
    .agg(
        recency_days=("donation_date", lambda x: (ref_date - x.max()).days),
        frequency=("donation_date", "count"),
        monetary_total=("amount", "sum"),
        monetary_avg=("amount", "mean"),
        first_date=("donation_date", "min"),
        last_date=("donation_date", "max"),
    )
    .reset_index()
)

rfm["span_days"] = (rfm["last_date"] - rfm["first_date"]).dt.days.clip(lower=0)

# Step C: Transforming and Scaling
# Apply log-transformation to reduce skew in frequency and monetary features
features = rfm[["recency_days", "frequency", "monetary_total"]].copy()

# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
features["frequency"] = np.log1p(features["frequency"])
features["monetary_total"] = np.log1p(features["monetary_total"])

scaler = StandardScaler()
X = scaler.fit_transform(features)

# Step D: choosing K and fitting KMeans
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 8, 4)

if len(rfm) < k:
    st.warning(
        f"Zu wenige Spender ({len(rfm)}) für k={k}. Bitte k reduzieren.")
    st.stop()

km = KMeans(n_clusters=k, random_state=42, n_init="auto")
rfm["cluster"] = km.fit_predict(X)

# Step E: Cluster summary and segment-names
summary = (
    rfm.groupby("cluster")
    .agg(
        donors=("donor_id", "count"),
        recency_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary_total", "mean"),
    )
    .reset_index()
)

# rounding values
summary[["recency_mean", "frequency_mean", "monetary_mean"]] = (
    summary[["recency_mean", "frequency_mean", "monetary_mean"]].round(1)
)

# Compute quantile thresholds for segmentation
# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
rec_q = summary["recency_mean"].quantile([0.33, 0.67])
freq_q = summary["frequency_mean"].quantile([0.33, 0.67])
mon_q = summary["monetary_mean"].quantile([0.33, 0.67])

# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
def label_cluster(row):
    # Champions: very recently, very frequently, very much
    if (row["recency_mean"] <= rec_q.loc[0.33] and
        row["frequency_mean"] >= freq_q.loc[0.67] and
            row["monetary_mean"] >= mon_q.loc[0.67]):
        return "Champions / Core Supporters"

    # Recent one-timers: recently, but rarely and little
    if (row["recency_mean"] <= rec_q.loc[0.33] and
            row["frequency_mean"] <= freq_q.loc[0.33]):
        return "Recent One-Timers"

    # Lapsed big donors: long time ago, but a large amount
    if (row["recency_mean"] >= rec_q.loc[0.67] and
            row["monetary_mean"] >= mon_q.loc[0.67]):
        return "Lapsed Big Donors"

    # Lost donors: long time ago, rarely
    if (row["recency_mean"] >= rec_q.loc[0.67] and
            row["frequency_mean"] <= freq_q.loc[0.33]):
        return "Lost Donors"

    return "Potential Loyalists"


summary["segment"] = summary.apply(label_cluster, axis=1)
rfm = rfm.merge(summary[["cluster", "segment"]], on="cluster", how="left")

# Step F: view and visualize
st.subheader("Cluster-Overview")
summary_display = summary.sort_values("segment").rename(columns={
    "cluster": "Cluster",
    "donors": "Donors",
    "recency_mean": "Avg. recency (days)",
    "frequency_mean": "Avg. frequency",
    "monetary_mean": "Avg. total amount",
    "segment": "Segment",
})

st.dataframe(summary_display)

# Project scaled features into 2D using PCA for visualization
# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

plot_df = pd.DataFrame(X2, columns=["PC1", "PC2"])
plot_df["cluster"] = rfm["cluster"]
plot_df["segment"] = rfm["segment"]

st.subheader("Cluster-Map (PCA)")
st.scatter_chart(plot_df, x="PC1", y="PC2", color="cluster")

st.subheader("Cluster-Sizes")
cluster_sizes = rfm["segment"].value_counts().reset_index()
cluster_sizes.columns = ["segment", "count"]
st.bar_chart(cluster_sizes, x="segment", y="count")

# Step G: Target List (Business Output)
st.subheader("Target-List for Outreach")

default_targets = [s for s in ["Champions / Core Supporters", "Potential Loyalists"]
                   if s in rfm["segment"].unique()]

target_segments = st.multiselect(
    "Which segments should be displayed?",
    options=sorted(rfm["segment"].unique()),
    default=default_targets
)

# Rank donors for outreach: recent first, then high frequency and amount
# NOTE: Parts of this implementation were developed with assistance from OpenAI ChatGPT (Dec 2025)
targets = rfm[rfm["segment"].isin(target_segments)].copy()
targets = targets.sort_values(
    ["recency_days", "frequency", "monetary_total"],
    ascending=[True, False, False]
)

# Merge names
if "first_name" in targets.columns and "last_name" in targets.columns:
    targets["name"] = targets["first_name"] + " " + targets["last_name"]
else:
    targets["name"] = targets["donor_id"]   # fallback


targets_display = targets[[
    "name", "segment", "recency_days",
    "frequency", "monetary_total", "monetary_avg", "span_days"
]].rename(columns={
    "name": "Donor",
    "segment": "Segment",
    "recency_days": "Recency (days)",
    "frequency": "Frequency",
    "monetary_total": "Total amount",
    "monetary_avg": "Average amount",
    "span_days": "Donation span (days)",
})

st.dataframe(targets_display)

st.info(
    "Interpretation: low recency + high frequency = very likely to donate again."
    " Prioritize these individuals (thank-you email, card, personal contact)."
)
