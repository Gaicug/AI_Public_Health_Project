import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Vulnerability Prediction",
    page_icon="🤖",
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

st.title("🤖 AI Vulnerability Prediction Engine")

st.write(
    "Enter public health indicators below and allow the "
    "AI model to predict the vulnerability level."
)

st.divider()


# ==============================
# INPUT SECTION
# ==============================

st.subheader("📋 Enter Public Health Indicators")

col1, col2 = st.columns(2)


with col1:

    population = st.number_input(
        "Population",
        value=int(df["Population"].mean())
    )

    population_density = st.number_input(
        "Population Density",
        value=float(df["Population_Density"].mean())
    )

    poverty_rate = st.number_input(
        "Poverty Rate (%)",
        value=float(df["Poverty_Rate"].mean())
    )

    health_facilities = st.number_input(
        "Health Facilities",
        value=int(df["Health_Facilities"].mean())
    )

    doctors = st.number_input(
        "Doctors per 10,000",
        value=float(df["Doctors_per_10000"].mean())
    )

    immunization = st.number_input(
        "Immunization Coverage (%)",
        value=float(df["Immunization_Coverage"].mean())
    )

    water = st.number_input(
        "Clean Water Access (%)",
        value=float(df["Clean_Water_Access"].mean())
    )

    maternal = st.number_input(
        "Maternal Mortality",
        value=float(df["Maternal_Mortality"].mean())
    )


with col2:

    under5 = st.number_input(
        "Under-5 Mortality",
        value=float(df["Under5_Mortality"].mean())
    )

    malnutrition = st.number_input(
        "Malnutrition Rate (%)",
        value=float(df["Malnutrition_Rate"].mean())
    )

    hiv = st.number_input(
        "HIV Prevalence (%)",
        value=float(df["HIV_Prevalence"].mean())
    )

    malaria = st.number_input(
        "Malaria Prevalence (%)",
        value=float(df["Malaria_Prevalence"].mean())
    )

    unemployment = st.number_input(
        "Unemployment Rate (%)",
        value=float(df["Unemployment_Rate"].mean())
    )

    latitude = st.number_input(
        "Latitude",
        value=float(df["Latitude"].mean()),
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        value=float(df["Longitude"].mean()),
        format="%.4f"
    )

    vulnerability_score = st.number_input(
        "Vulnerability Score",
        value=float(df["Vulnerability_Score"].mean())
    )


st.divider()


# ==============================
# AI PREDICTION BUTTON
# ==============================

if st.button(
    "🤖 RUN AI VULNERABILITY PREDICTION",
    use_container_width=True
):

    # Create input data
    input_data = pd.DataFrame({

        "Population": [population],
        "Population_Density": [population_density],
        "Poverty_Rate": [poverty_rate],
        "Health_Facilities": [health_facilities],
        "Doctors_per_10000": [doctors],
        "Immunization_Coverage": [immunization],
        "Clean_Water_Access": [water],
        "Maternal_Mortality": [maternal],
        "Under5_Mortality": [under5],
        "Malnutrition_Rate": [malnutrition],
        "HIV_Prevalence": [hiv],
        "Malaria_Prevalence": [malaria],
        "Unemployment_Rate": [unemployment],
        "Vulnerability_Score": [vulnerability_score],
        "Latitude": [latitude],
        "Longitude": [longitude]

    })

    # Arrange columns in the exact order
    # expected by the trained AI model
    input_data = input_data[
        model.feature_names_in_
    ]

    with st.spinner(
        "🤖 AI model is analyzing the public health indicators..."
    ):

        prediction = model.predict(
            input_data
        )

        probabilities = model.predict_proba(
            input_data
        )

    # Convert encoded prediction
    result = encoder.inverse_transform(
        prediction
    )

    # AI confidence
    confidence = probabilities.max() * 100


    # ==============================
    # DISPLAY RESULT
    # ==============================

    st.divider()

    st.subheader("🤖 AI Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if result[0] == "High":

            st.error(
                "🔴 HIGH VULNERABILITY"
            )

        elif result[0] == "Medium":

            st.warning(
                "🟠 MEDIUM VULNERABILITY"
            )

        else:

            st.success(
                "🟢 LOW VULNERABILITY"
            )


    with result_col2:

        st.metric(
            "🎯 AI Prediction Confidence",
            f"{confidence:.2f}%"
        )


    # ==============================
    # AI DECISION SUPPORT
    # ==============================

    st.divider()

    st.subheader(
        "🧠 AI Decision Support Recommendations"
    )

    recommendations = []

    if poverty_rate > df["Poverty_Rate"].mean():

        recommendations.append(
            "Prioritize poverty reduction and community health interventions."
        )

    if immunization < df["Immunization_Coverage"].mean():

        recommendations.append(
            "Strengthen immunization outreach programmes."
        )

    if health_facilities < df["Health_Facilities"].mean():

        recommendations.append(
            "Increase healthcare infrastructure and facility capacity."
        )

    if water < df["Clean_Water_Access"].mean():

        recommendations.append(
            "Improve access to clean water and sanitation services."
        )

    if under5 > df["Under5_Mortality"].mean():

        recommendations.append(
            "Prioritize child health and under-five mortality interventions."
        )


    # General recommendation based on AI result

    if result[0] == "High":

        recommendations.append(
            "🚨 Immediate intervention is recommended due to high predicted vulnerability."
        )

    elif result[0] == "Medium":

        recommendations.append(
            "⚠️ Strengthen preventive healthcare and disease monitoring."
        )

    else:

        recommendations.append(
            "✅ Maintain existing interventions and continue monitoring."
        )


    for recommendation in recommendations:

        st.write(f"• {recommendation}")


# ==============================
# AI MODEL INFORMATION
# ==============================

st.divider()

st.subheader("🧠 About the AI Model")

st.info(
    """
This system uses a trained Machine Learning model to analyze
multiple public health, socioeconomic, healthcare and geographic
indicators.

The AI model predicts vulnerability into three categories:

🔴 High Vulnerability

🟠 Medium Vulnerability

🟢 Low Vulnerability
"""
)

st.caption(
    "This application is a public health decision-support and research prototype."
)