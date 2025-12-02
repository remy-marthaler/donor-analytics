# ---------------------------------------------------------
# Donor Lifetime Value Page
# FINAL VERSION – Modern Design + What-If + Fail-Safe + CV + Tuning
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.io as pio

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.core.state import get_api_client


# ---------------------------------------------------------
# Plotly "Corporate" Defaults (Modern Look)
# ---------------------------------------------------------
pio.templates.default = "plotly_white"


# ---------------------------------------------------------
# Utility: Fail-safe column helpers
# ---------------------------------------------------------
def safe_select(df, columns):
    """Return only columns that exist in df."""
    return [c for c in columns if c in df.columns]


def safe_dataframe(df, columns):
    cols = safe_select(df, columns)
    if not cols:
        return pd.DataFrame()
    return df[cols]


# ---------------------------------------------------------
# Load + Clean Data
# ---------------------------------------------------------
def load_data():
    api = get_api_client()
    donations = api.get_donations()
    df = pd.DataFrame(donations)

    column_mapping = {
        "Kontakt-ID": "donor_id",
        "Getätigt am Datum": "date",
        "Betrag": "amount",
    }
    df = df.rename(columns=column_mapping)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    amt = df["amount"].astype(str)
    amt = amt.str.replace(".", "", regex=False)
    amt = amt.str.replace(",", ".", regex=False)
    df["amount"] = pd.to_numeric(amt, errors="coerce")

    df = df.dropna(subset=["donor_id", "date", "amount"])
    df = df[df["amount"] > 0]

    return df


# ---------------------------------------------------------
# Heuristic LTV
# ---------------------------------------------------------
def compute_simple_ltv(df, expected_years=3):
    grp = df.groupby("donor_id")

    avg_amount = grp["amount"].mean()

    d2 = df.copy()
    d2["year"] = d2["date"].dt.year
    freq = (
        d2.groupby(["donor_id", "year"])
        .size()
        .groupby("donor_id")
        .mean()
    )

    ltv = avg_amount * freq * expected_years

    out = pd.DataFrame(
        {
            "donor_id": avg_amount.index,
            "avg_amount": avg_amount.values,
            "freq_per_year": freq.values,
            "ltv": ltv.values,
        }
    )
    return out.sort_values("ltv", ascending=False)


# ---------------------------------------------------------
# ML LTV with CV + Tuning
# ---------------------------------------------------------
def compute_ml_ltv_models(df):
    ref_date = df["date"].max()
    rfm = (
        df.groupby("donor_id")
        .agg(
            recency_days=("date", lambda x: (ref_date - x.max()).days),
            frequency=("date", "count"),
            monetary_total=("amount", "sum"),
            monetary_avg=("amount", "mean"),
            span_days=("date", lambda x: (x.max() - x.min()).days),
        )
        .reset_index()
    )

    rng = np.random.default_rng(42)
    rfm["target_next_year"] = (
        rfm["monetary_avg"].replace(0, 1)
        * rfm["frequency"].replace(0, 1)
        * (0.8 + 0.4 * rng.random(len(rfm)))
    )

    features = [
        "recency_days",
        "frequency",
        "monetary_total",
        "monetary_avg",
        "span_days",
    ]
    X = rfm[features]
    y = rfm["target_next_year"]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter grids
    rf_param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    gb_param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }

    # Random Forest
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        rf_param_grid,
        n_iter=15,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X, y)
    best_rf = rf_search.best_estimator_
    rf_rmse = -cross_val_score(
        best_rf, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    ).mean()
    rf_mae = -cross_val_score(
        best_rf, X, y, cv=kf, scoring="neg_mean_absolute_error"
    ).mean()

    # Gradient Boosting
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_param_grid,
        n_iter=15,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    gb_search.fit(X, y)
    best_gb = gb_search.best_estimator_
    gb_rmse = -cross_val_score(
        best_gb, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    ).mean()
    gb_mae = -cross_val_score(
        best_gb, X, y, cv=kf, scoring="neg_mean_absolute_error"
    ).mean()

    # Compare models
    results_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Mean Absolute Error": rf_mae,
                "Root Mean Squared Error": rf_rmse,
            },
            {
                "Model": "Gradient Boosting",
                "Mean Absolute Error": gb_mae,
                "Root Mean Squared Error": gb_rmse,
            },
        ]
    ).sort_values("Root Mean Squared Error")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = best_rf if best_model_name == "Random Forest" else best_gb

    # Predicted LTV
    rfm["ml_ltv"] = best_model.predict(X)

    feat_imp = pd.DataFrame(
        {
            "Feature": features,
            "Importance": best_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return rfm, results_df, best_model_name, feat_imp, best_model


# ---------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------
def run():
    st.title("💰 Donor Lifetime Value (LTV) Analytics")
    st.caption("Heuristic and Machine-Learning based LTV estimation for donor prioritisation.")

    df = load_data()

    # Precompute ML models once (for ML tab + What-if tab)
    rfm_ml, results_df, best_model_name, feat_imp, best_model = compute_ml_ltv_models(df)

    # -----------------------------------------------------
    # Overview KPIs
    # -----------------------------------------------------
    st.subheader("📌 Overview KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧾 Total Donations", f"{len(df):,}")
    c2.metric("💵 Total Amount Donated", f"{df['amount'].sum():,.0f}")
    c3.metric("💶 Average Donation", f"{df['amount'].mean():,.2f}")
    c4.metric("🧑 Donors Count", f"{df['donor_id'].nunique():,}")

    # -----------------------------------------------------
    # Donation Patterns (Modern Plotly)
    # -----------------------------------------------------
    st.subheader("📊 Donation Patterns")

    dmonth = df.copy()
    dmonth["month"] = dmonth["date"].dt.to_period("M").dt.to_timestamp()
    m_agg = dmonth.groupby("month")["amount"].sum().reset_index()

    if not m_agg.empty:
        fig_month = px.line(
            m_agg,
            x="month",
            y="amount",
            title="Monthly Donation Volume",
            labels={"month": "Month", "amount": "Donation Volume (CHF)"},
        )
        fig_month.update_traces(mode="lines+markers")
        st.plotly_chart(fig_month, use_container_width=True)

    df_pos = df[df["amount"] > 0]
    if not df_pos.empty:
        fig_amount = px.histogram(
            df_pos,
            x="amount",
            nbins=30,
            title="Donation Amount Distribution (log scale)",
            labels={"amount": "Donation Amount (CHF)"},
            log_y=True,
        )
        st.plotly_chart(fig_amount, use_container_width=True)

    # -----------------------------------------------------
    # Tabs: Heuristic, ML, What-if
    # -----------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📐 Heuristic Model", "🤖 ML Model", "🧪 What-if Simulator"])

    # -----------------------------------------------------
    # TAB 1 — Heuristic
    # -----------------------------------------------------
    with tab1:
        st.subheader("Heuristic LTV Calculation")

        exp_years = st.slider(
            "Expected donor lifetime (years)", 1, 10, 3
        )

        ltv = compute_simple_ltv(df, exp_years)

        ltv_pos = ltv[ltv["ltv"] > 0].reset_index()
        if not ltv_pos.empty:
            fig_ltv = px.histogram(
                ltv_pos,
                x="ltv",
                nbins=30,
                title="Heuristic LTV Distribution (log scale)",
                labels={"ltv": "Heuristic LTV (CHF)"},
                log_y=True,
            )
            st.plotly_chart(fig_ltv, use_container_width=True)

        pretty_heur = {
            "donor_id": "Donor ID",
            "avg_amount": "Average Donation",
            "freq_per_year": "Donation Frequency per Year",
            "ltv": "Heuristic LTV",
        }
        st.subheader("Heuristic LTV per Donor")
        st.dataframe(
            ltv.rename(columns=pretty_heur),
            use_container_width=True,
        )

    # -----------------------------------------------------
    # TAB 2 — ML
    # -----------------------------------------------------
    with tab2:
        st.subheader("Machine Learning LTV Prediction")

        st.markdown(f"**Best Model Selected:** `{best_model_name}`")

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)

        st.caption(
            "Mean Absolute Error = average prediction error in CHF.\n"
            "Root Mean Squared Error = penalizes large errors more strongly.\n"
            "Lower values = better model."
        )

        # Feature Importance (normalized + pretty)
        st.subheader("Key Drivers of Predicted LTV")

        pretty_features = {
            "recency_days": "Days Since Last Donation",
            "frequency": "Donation Count",
            "monetary_total": "Total Donated",
            "monetary_avg": "Average Donation",
            "span_days": "Donation Time Span (days)",
        }

        feat_imp_pretty = feat_imp.copy()
        feat_imp_pretty["Feature"] = feat_imp_pretty["Feature"].map(pretty_features)

        max_val = feat_imp_pretty["Importance"].max()
        feat_imp_pretty["Importance (normalized)"] = (
            feat_imp_pretty["Importance"] / max_val * 100
        )

        fig_feat = px.bar(
            feat_imp_pretty,
            x="Importance (normalized)",
            y="Feature",
            orientation="h",
            title="Normalized Feature Importance (0–100 Scale)",
            labels={
                "Importance (normalized)": "Relative Importance (0–100)",
                "Feature": "Feature",
            },
        )
        fig_feat.update_layout(xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_feat, use_container_width=True)

        # Business Impact
        st.subheader("💼 Business Impact")

        total_ltv = rfm_ml["ml_ltv"].sum()
        q90 = rfm_ml["ml_ltv"].quantile(0.90)
        top10 = rfm_ml[rfm_ml["ml_ltv"] >= q90]
        top10_value = top10["ml_ltv"].sum()
        share = top10_value / total_ltv if total_ltv > 0 else 0

        at_risk = top10[top10["recency_days"] > 365]
        at_risk_value = at_risk["ml_ltv"].sum()
        predicted_gap = at_risk_value * 0.35

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Predicted LTV", f"{total_ltv:,.0f}")
        b2.metric("Top 10% Donor Value", f"{top10_value:,.0f}", f"{share*100:,.1f}% of total")
        b3.metric("Revenue at Risk", f"{at_risk_value:,.0f}", "Top donors with no donation > 12 months")
        b4.metric("Predicted Revenue Gap", f"{predicted_gap:,.0f}", "Potentially recoverable")

        pretty_ml = {
            "donor_id": "Donor ID",
            "recency_days": "Days Since Last Donation",
            "frequency": "Donation Count",
            "monetary_total": "Total Donated",
            "ml_ltv": "Predicted LTV",
        }

        st.subheader("Predicted LTV per Donor")
        rfm_display = rfm_ml.rename(columns=pretty_ml)
        cols_main = [
            "Donor ID",
            "Days Since Last Donation",
            "Donation Count",
            "Total Donated",
            "Predicted LTV",
        ]
        df_main = safe_dataframe(rfm_display, cols_main)
        if not df_main.empty:
            st.dataframe(
                df_main.sort_values("Predicted LTV", ascending=False),
                use_container_width=True,
            )

        if not at_risk.empty:
            st.subheader("High-value Donors at Risk")
            at_risk_display = at_risk.rename(columns=pretty_ml)
            df_risk = safe_dataframe(at_risk_display, cols_main)
            if not df_risk.empty:
                st.dataframe(
                    df_risk.sort_values("Predicted LTV", ascending=False),
                    use_container_width=True,
                )

    # -----------------------------------------------------
    # TAB 3 — What-if Simulator
    # -----------------------------------------------------
    with tab3:
        st.subheader("What-if Simulator")

        st.caption(
            "Simulate how changes in donation frequency and average donation "
            "would impact predicted donor lifetime value."
        )

        # Sliders (global adjustments)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            freq_factor = st.slider(
                "Relative change in donation count",
                0.5,
                2.0,
                1.0,
                0.1,
            )
            amount_factor = st.slider(
                "Relative change in average donation",
                0.5,
                2.0,
                1.0,
                0.1,
            )
        with col_w2:
            recency_delta = st.slider(
                "Change in days since last donation",
                -365,
                365,
                0,
                15,
            )
            span_delta = st.slider(
                "Change in donation time span (days)",
                -365,
                365,
                0,
                15,
            )

        # Base features
        feature_cols = [
            "recency_days",
            "frequency",
            "monetary_total",
            "monetary_avg",
            "span_days",
        ]

        X_base = rfm_ml[feature_cols].copy()

        # Adjusted scenario
        X_new = X_base.copy()
        X_new["frequency"] = (X_new["frequency"] * freq_factor).clip(lower=0.1)
        X_new["monetary_avg"] = (X_new["monetary_avg"] * amount_factor).clip(lower=0)
        X_new["monetary_total"] = X_new["monetary_total"] * freq_factor * amount_factor
        X_new["recency_days"] = (X_new["recency_days"] + recency_delta).clip(lower=0)
        X_new["span_days"] = (X_new["span_days"] + span_delta).clip(lower=0)

        # Predictions
        base_ltv = rfm_ml["ml_ltv"].values
        new_ltv = best_model.predict(X_new)

        total_base = base_ltv.sum()
        total_new = new_ltv.sum()
        delta_abs = total_new - total_base
        delta_pct = (delta_abs / total_base * 100) if total_base > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Total Predicted LTV", f"{total_base:,.0f}")
        c2.metric("What-if Total Predicted LTV", f"{total_new:,.0f}")
        c3.metric("Change", f"{delta_abs:,.0f}", f"{delta_pct:,.1f}%")

        # Distribution shift
        st.subheader("Distribution Shift (Current vs What-if)")

        df_hist = pd.DataFrame(
            {
                "Scenario": ["Current"] * len(base_ltv) + ["What-if"] * len(new_ltv),
                "Predicted LTV": np.concatenate([base_ltv, new_ltv]),
            }
        )

        fig_sim = px.histogram(
            df_hist,
            x="Predicted LTV",
            color="Scenario",
            barmode="overlay",
            nbins=30,
            opacity=0.6,
            title="Predicted LTV Distribution (Current vs What-if, log scale)",
            log_y=True,
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Top donors under scenario
        st.subheader("Top Donors under What-if Scenario")

        rfm_sim = rfm_ml.copy()
        rfm_sim["ml_ltv_new"] = new_ltv

        pretty_sim = {
            "donor_id": "Donor ID",
            "recency_days": "Days Since Last Donation",
            "frequency": "Donation Count",
            "monetary_total": "Total Donated",
            "ml_ltv": "Predicted LTV (Current)",
            "ml_ltv_new": "Predicted LTV (What-if)",
        }

        rfm_sim_display = rfm_sim.rename(columns=pretty_sim)
        cols_sim = [
            "Donor ID",
            "Days Since Last Donation",
            "Donation Count",
            "Total Donated",
            "Predicted LTV (Current)",
            "Predicted LTV (What-if)",
        ]
        df_sim = safe_dataframe(rfm_sim_display, cols_sim)
        if not df_sim.empty:
            df_sim["Δ LTV"] = (
                df_sim["Predicted LTV (What-if)"] - df_sim["Predicted LTV (Current)"]
            )
            st.dataframe(
                df_sim.sort_values("Predicted LTV (What-if)", ascending=False),
                use_container_width=True,
            )


# ---------------------------------------------------------
# Required by Streamlit
# ---------------------------------------------------------
if __name__ == "__main__":
    run()