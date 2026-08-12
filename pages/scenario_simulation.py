import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Scenario Simulation",
    page_icon="🔮",
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

st.title("🔮 AI Public Health Scenario Simulation")

st.markdown("""
This AI simulation allows users to test **what-if scenarios**.

Select a county, modify important public health indicators, and compare
the original AI prediction with a simulated intervention scenario.

Example:

- What if immunization coverage improves?
- What if more health facilities are added?
- What if clean water access increases?
- What if poverty levels decrease?

The Machine Learning model will analyze the new scenario and generate
an updated vulnerability prediction.
""")

st.divider()


# ==============================
# SELECT COUNTY
# ==============================

st.subheader("🏥 Step 1: Select a County")

selected_county = st.selectbox(
    "Select County for Simulation",
    df["County"].unique()
)

county_data = df[
    df["County"] == selected_county
].iloc[0]

st.success(
    f"Selected County: {selected_county}"
)


# ==============================
# ORIGINAL COUNTY INDICATORS
# ==============================

st.subheader("📊 Current Public Health Situation")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Vulnerability",
    county_data["Vulnerability_Level"]
)

col2.metric(
    "Vulnerability Score",
    f"{county_data['Vulnerability_Score']:.2f}"
)

col3.metric(
    "Poverty Rate",
    f"{county_data['Poverty_Rate']:.1f}%"
)

col4.metric(
    "Immunization",
    f"{county_data['Immunization_Coverage']:.1f}%"
)

st.divider()


# ==============================
# INTERVENTION SIMULATION
# ==============================

st.subheader("🧪 Step 2: Simulate Public Health Interventions")

st.write(
    "Adjust the indicators below to simulate possible public health improvements."
)

col1, col2 = st.columns(2)

with col1:

    poverty_rate = st.slider(
        "Poverty Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Poverty_Rate"])
    )

    health_facilities = st.slider(
        "Health Facilities",
        min_value=0,
        max_value=500,
        value=int(county_data["Health_Facilities"])
    )

    doctors = st.slider(
        "Doctors per 10,000",
        min_value=0.0,
        max_value=50.0,
        value=float(county_data["Doctors_per_10000"])
    )

    immunization = st.slider(
        "Immunization Coverage (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Immunization_Coverage"])
    )

    water = st.slider(
        "Clean Water Access (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Clean_Water_Access"])
    )

    unemployment = st.slider(
        "Unemployment Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Unemployment_Rate"])
    )


with col2:

    maternal = st.slider(
        "Maternal Mortality",
        min_value=0.0,
        max_value=1000.0,
        value=float(county_data["Maternal_Mortality"])
    )

    under5 = st.slider(
        "Under-5 Mortality",
        min_value=0.0,
        max_value=200.0,
        value=float(county_data["Under5_Mortality"])
    )

    malnutrition = st.slider(
        "Malnutrition Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Malnutrition_Rate"])
    )

    hiv = st.slider(
        "HIV Prevalence (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["HIV_Prevalence"])
    )

    malaria = st.slider(
        "Malaria Prevalence (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(county_data["Malaria_Prevalence"])
    )


st.divider()


# ==============================
# RUN AI SIMULATION
# ==============================

if st.button(
    "🔮 RUN AI SCENARIO SIMULATION",
    use_container_width=True
):

    # Create simulated data
    simulation_data = pd.DataFrame({

        "Population": [
            county_data["Population"]
        ],

        "Population_Density": [
            county_data["Population_Density"]
        ],

        "Poverty_Rate": [
            poverty_rate
        ],

        "Health_Facilities": [
            health_facilities
        ],

        "Doctors_per_10000": [
            doctors
        ],

        "Immunization_Coverage": [
            immunization
        ],

        "Clean_Water_Access": [
            water
        ],

        "Maternal_Mortality": [
            maternal
        ],

        "Under5_Mortality": [
            under5
        ],

        "Malnutrition_Rate": [
            malnutrition
        ],

        "HIV_Prevalence": [
            hiv
        ],

        "Malaria_Prevalence": [
            malaria
        ],

        "Unemployment_Rate": [
            unemployment
        ],

        "Vulnerability_Score": [
            county_data["Vulnerability_Score"]
        ],

        "Latitude": [
            county_data["Latitude"]
        ],

        "Longitude": [
            county_data["Longitude"]
        ]

    })

    # Arrange columns in EXACT model order
    simulation_data = simulation_data[
        model.feature_names_in_
    ]

    # ==============================
    # AI PREDICTION
    # ==============================

    with st.spinner(
        "🤖 AI is analyzing the simulated scenario..."
    ):

        prediction = model.predict(
            simulation_data
        )

        probabilities = model.predict_proba(
            simulation_data
        )

    result = encoder.inverse_transform(
        prediction
    )

    confidence = probabilities.max() * 100


    # ==============================
    # DISPLAY RESULTS
    # ==============================

    st.divider()

    st.subheader("🤖 AI Scenario Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Original Vulnerability",
        county_data["Vulnerability_Level"]
    )

    result_col2.metric(
        "Simulated AI Prediction",
        result[0]
    )

    result_col3.metric(
        "AI Confidence",
        f"{confidence:.2f}%"
    )


    # ==============================
    # SCENARIO INTERPRETATION
    # ==============================

    st.divider()

    st.subheader("🧠 AI Scenario Interpretation")

    original = county_data["Vulnerability_Level"]
    simulated = result[0]

    if original == simulated:

        st.info(
            f"""
The AI prediction remains **{simulated} Vulnerability**.

The simulated changes may not yet be sufficient to significantly
change the county's predicted vulnerability classification.
"""
        )

    else:

        st.success(
            f"""
The AI model predicts a change from **{original} Vulnerability**
to **{simulated} Vulnerability**.

This suggests that the simulated intervention changes could influence
the public health vulnerability classification.
"""
        )


    # ==============================
    # INTERVENTION SUMMARY
    # ==============================

    st.divider()

    st.subheader("📋 Simulated Intervention Summary")

    comparison_data = pd.DataFrame({

        "Indicator": [
            "Poverty Rate",
            "Health Facilities",
            "Doctors per 10,000",
            "Immunization Coverage",
            "Clean Water Access",
            "Unemployment Rate"
        ],

        "Original": [
            county_data["Poverty_Rate"],
            county_data["Health_Facilities"],
            county_data["Doctors_per_10000"],
            county_data["Immunization_Coverage"],
            county_data["Clean_Water_Access"],
            county_data["Unemployment_Rate"]
        ],

        "Simulated": [
            poverty_rate,
            health_facilities,
            doctors,
            immunization,
            water,
            unemployment
        ]

    })

    comparison_data["Change"] = (
        comparison_data["Simulated"]
        - comparison_data["Original"]
    )

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    st.divider()


    # ==============================
    # AI RECOMMENDATION
    # ==============================

    st.subheader("🎯 AI-Assisted Decision Support")

    if result[0] == "High":

        st.error("""
🚨 **High Vulnerability Predicted**

Recommended actions:

- Increase healthcare resources.
- Expand access to healthcare facilities.
- Improve immunization coverage.
- Strengthen disease surveillance.
- Improve clean water and sanitation.
- Prioritize vulnerable populations.
""")

    elif result[0] == "Medium":

        st.warning("""
⚠️ **Medium Vulnerability Predicted**

Recommended actions:

- Strengthen preventive healthcare.
- Improve community health programmes.
- Monitor key health indicators.
- Expand immunization outreach.
""")

    else:

        st.success("""
🟢 **Low Vulnerability Predicted**

Recommended actions:

- Maintain existing interventions.
- Continue monitoring public health indicators.
- Sustain healthcare investment.
- Strengthen preparedness for future health risks.
""")


# ==============================
# FOOTER
# ==============================

st.divider()

st.info("""
⚠️ **Research and Decision-Support Notice**

The scenario simulation uses the trained Machine Learning model to
explore potential changes in vulnerability classification.

Results should be interpreted as AI-assisted decision support and
not as definitive forecasts or clinical recommendations.
""")

st.caption(
    "AI-Driven Public Health Vulnerability Index and Decision Support System for Kenyan Counties | ENGAGE Project 2026"
)