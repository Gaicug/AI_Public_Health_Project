import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/kenya_public_health_vulnerability.csv"
    )


# ==============================
# LOAD AI MODEL
# ==============================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/vulnerability_model.pkl"
    )


df = load_data()
model = load_model()


# ==============================
# PAGE TITLE
# ==============================

st.title("🧠 Explainable AI")

st.markdown("""
### Understanding Why the AI Makes Its Predictions

This page explains the factors used by the Machine Learning model when
predicting public health vulnerability.

Explainable AI improves transparency by showing which public health,
healthcare, socioeconomic and geographic indicators have the greatest
influence on the model.
""")

st.divider()


# ==============================
# CHECK FEATURE IMPORTANCE
# ==============================

if hasattr(model, "feature_importances_"):

    feature_names = model.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    # ==============================
    # FEATURE IMPORTANCE CHART
    # ==============================

    st.subheader("📊 AI Feature Importance")

    st.write(
        "The chart below shows how strongly each feature contributes "
        "to the AI model's vulnerability predictions."
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Public Health Factors Influencing AI Predictions"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()


    # ==============================
    # TOP AI DRIVERS
    # ==============================

    st.subheader("🔍 Top AI Vulnerability Drivers")

    top_features = importance_df.head(5)

    col1, col2, col3, col4, col5 = st.columns(5)

    columns = [
        col1,
        col2,
        col3,
        col4,
        col5
    ]

    for i, (_, row) in enumerate(
        top_features.iterrows()
    ):

        columns[i].metric(
            f"#{i + 1}",
            row["Feature"],
            f"{row['Importance'] * 100:.1f}%"
        )

    st.divider()


    # ==============================
    # FEATURE IMPORTANCE TABLE
    # ==============================

    st.subheader("📋 Detailed AI Feature Importance")

    display_df = importance_df.copy()

    display_df["Importance (%)"] = (
        display_df["Importance"] * 100
    ).round(2)

    display_df = display_df[
        [
            "Feature",
            "Importance (%)"
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.divider()


    # ==============================
    # AI EXPLANATION
    # ==============================

    st.subheader("🤖 How to Interpret the AI Results")

    st.info("""
A higher feature importance means that the Machine Learning model relied
more heavily on that indicator when separating counties into different
vulnerability categories.

For example:

• A highly important Poverty Rate suggests that socioeconomic conditions
may strongly influence vulnerability classification.

• A highly important Immunization Coverage indicator suggests that
vaccination access plays an important role in the model's predictions.

• A highly important Under-5 Mortality indicator suggests that child
health outcomes are strongly associated with vulnerability.

Feature importance describes the model's behaviour. It does not, by
itself, prove that one factor directly causes public health vulnerability.
""")

    st.divider()


    # ==============================
    # TOP 10 FEATURES
    # ==============================

    st.subheader("🏆 Top 10 Most Influential AI Features")

    top10 = importance_df.head(10).copy()

    top10.insert(
        0,
        "Rank",
        range(1, len(top10) + 1)
    )

    top10["Importance (%)"] = (
        top10["Importance"] * 100
    ).round(2)

    st.dataframe(
        top10[
            [
                "Rank",
                "Feature",
                "Importance (%)"
            ]
        ],
        use_container_width=True
    )

else:

    st.error(
        """
The loaded model does not provide feature importance values.

This Explainable AI page requires a model such as Random Forest,
Decision Tree or another estimator that supports feature_importances_.
"""
    )


# ==============================
# PROJECT AI TRANSPARENCY
# ==============================

st.divider()

st.subheader("🔐 AI Transparency")

st.markdown("""
This project uses Explainable AI techniques to improve transparency and
interpretability.

The system allows users to understand:

🤖 Which indicators influence the model.

📊 How strongly each feature contributes to predictions.

🏥 Which public health factors the model considers most important.

⚠️ Why AI predictions should be used as decision-support rather than
as a replacement for professional public health judgement.
""")

st.caption(
    "Feature importance represents the influence of variables within "
    "the trained Machine Learning model."
)