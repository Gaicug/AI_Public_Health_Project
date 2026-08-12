import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Model Performance",
    page_icon="📈",
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

    model = joblib.load(
        "models/vulnerability_model.pkl"
    )

    encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    return model, encoder


df = load_data()
model, encoder = load_model()


# ==============================
# PAGE TITLE
# ==============================

st.title("📈 AI Model Performance")

st.markdown("""
This page evaluates the performance of the Machine Learning model used
to predict public health vulnerability levels.

The evaluation demonstrates how well the AI model classifies counties
into:

🔴 High Vulnerability

🟠 Medium Vulnerability

🟢 Low Vulnerability
""")

st.divider()


# ==============================
# PREPARE MODEL DATA
# ==============================

feature_names = list(
    model.feature_names_in_
)

X = df[feature_names]

y_true = df["Vulnerability_Encoded"]

# ==============================
# MODEL PREDICTIONS
# ==============================

y_pred = model.predict(X)

accuracy = accuracy_score(
    y_true,
    y_pred
)

# ==============================
# KPI SECTION
# ==============================

st.subheader("🤖 AI Model Performance Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "🎯 Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "📊 Features Used",
    len(feature_names)
)

col3.metric(
    "🏥 Counties Evaluated",
    len(df)
)

st.divider()


# ==============================
# MODEL INFORMATION
# ==============================

st.subheader("🧠 AI Model Information")

model_name = type(model).__name__

st.info(
    f"""
**Machine Learning Model:** {model_name}

**Prediction Target:** Public Health Vulnerability Level

**Number of Features:** {len(feature_names)}

**Number of Records Evaluated:** {len(df)}
"""
)

st.divider()


# ==============================
# CONFUSION MATRIX
# ==============================




# ==============================
# CONFUSION MATRIX
# ==============================

st.subheader("🔍 Confusion Matrix")

st.write(
    "The confusion matrix compares the actual vulnerability levels "
    "with the vulnerability levels predicted by the AI model."
)

# Create confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred
)

# Convert class labels to a normal Python list
class_labels = encoder.inverse_transform(
    sorted(df["Vulnerability_Encoded"].unique())
).tolist()

# Create DataFrame for Plotly
cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

# Create interactive confusion matrix
fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    labels=dict(
        x="Predicted Vulnerability",
        y="Actual Vulnerability",
        color="Number of Counties"
    ),
    title="AI Model Confusion Matrix"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

st.divider()


# ==============================
# CLASSIFICATION REPORT
# ==============================

st.subheader("📋 Classification Report")

report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

st.divider()


# ==============================
# PREDICTION DISTRIBUTION
# ==============================

st.subheader("📊 AI Prediction Distribution")

prediction_labels = encoder.inverse_transform(
    y_pred
)

prediction_df = pd.DataFrame({
    "Predicted Vulnerability": prediction_labels
})

prediction_counts = (
    prediction_df[
        "Predicted Vulnerability"
    ]
    .value_counts()
    .reset_index()
)

prediction_counts.columns = [
    "Vulnerability Level",
    "Count"
]

fig = px.bar(
    prediction_counts,
    x="Vulnerability Level",
    y="Count",
    color="Vulnerability Level",
    title="Distribution of AI Predictions"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==============================
# MODEL FEATURES
# ==============================

st.subheader("📋 Features Used by the AI Model")

features_df = pd.DataFrame({
    "Feature": feature_names
})

st.dataframe(
    features_df,
    use_container_width=True
)

st.divider()


# ==============================
# MODEL PERFORMANCE INTERPRETATION
# ==============================

st.subheader("🧠 Performance Interpretation")

if accuracy >= 0.90:

    st.success(
        f"""
The AI model achieved an accuracy of **{accuracy * 100:.2f}%** on
the available project dataset.

This indicates that the model can successfully identify patterns in
the provided public health indicators.

However, performance on new real-world data should be independently
validated before operational use.
"""
    )

elif accuracy >= 0.70:

    st.warning(
        f"""
The AI model achieved an accuracy of **{accuracy * 100:.2f}%**.

The model demonstrates useful predictive capability, but further
training, validation and additional data could improve performance.
"""
    )

else:

    st.error(
        f"""
The AI model achieved an accuracy of **{accuracy * 100:.2f}%**.

The model may require additional training, feature engineering or
more representative data before reliable deployment.
"""
    )


# ==============================
# IMPORTANT NOTICE
# ==============================

st.divider()

st.subheader("⚠️ AI Evaluation Notice")

st.info("""
The performance metrics shown on this page evaluate the currently
loaded Machine Learning model using the available dataset.

For a rigorous research evaluation, the model should also be assessed
using a separate unseen test dataset or cross-validation.

This system is designed as an AI-assisted public health
decision-support and research prototype.
""")

st.caption(
    "AI-Driven Public Health Vulnerability Index and Decision Support System for Kenyan Counties | ENGAGE Project 2026"
)