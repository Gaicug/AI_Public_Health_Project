import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
#page navigation
# =========================================================
# SIDEBAR NAVIGATION
# =========================================================




# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Public Health Vulnerability Index",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    footer {
        visibility: hidden;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 20px;
        color: #555;
        margin-bottom: 25px;
    }

    .ai-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border-left: 5px solid #2e7d32;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/kenya_public_health_vulnerability.csv"
    )


# =========================================================
# LOAD MODEL
# =========================================================

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


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏥 AI-Driven Public Health Vulnerability Index</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Decision Support System for Kenyan Counties 🇰🇪</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    This AI-powered platform analyzes socioeconomic, healthcare,
    demographic and public health indicators to identify vulnerable
    counties and support evidence-based decision making.
    """
)


st.divider()


# =========================================================
# NATIONAL OVERVIEW
# =========================================================

st.subheader("🇰🇪 National Public Health Overview")


total_counties = len(df)

average_score = df[
    "Vulnerability_Score"
].mean()

average_poverty = df[
    "Poverty_Rate"
].mean()

average_immunization = df[
    "Immunization_Coverage"
].mean()


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "🏘️ Counties",
    total_counties
)

col2.metric(
    "🤖 Average Vulnerability",
    f"{average_score:.2f}"
)

col3.metric(
    "📉 Average Poverty",
    f"{average_poverty:.1f}%"
)

col4.metric(
    "💉 Average Immunization",
    f"{average_immunization:.1f}%"
)


st.divider()


# =========================================================
# AI SYSTEM DESCRIPTION
# =========================================================

st.subheader("🤖 What Does the AI System Do?")


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        <div class="ai-card">

        <h3>🧠 Machine Learning Prediction</h3>

        The trained Machine Learning model analyzes multiple
        public health indicators and predicts whether a county
        has High, Medium or Low vulnerability.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-card">

        <h3>🚨 Early Warning</h3>

        The system identifies counties with high vulnerability
        scores and highlights areas requiring priority attention.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-card">

        <h3>🔮 Scenario Simulation</h3>

        Users can change public health indicators and test
        possible intervention scenarios using the trained AI model.

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="ai-card">

        <h3>🧠 Explainable AI</h3>

        The system displays feature importance so users can
        understand which indicators influence the model's
        predictions.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-card">

        <h3>🗺️ Geographic Intelligence</h3>

        County vulnerability is displayed geographically using
        an interactive Kenya vulnerability map.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="ai-card">

        <h3>📊 Data Analytics</h3>

        Interactive statistical visualizations help identify
        relationships and patterns across public health indicators.

        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# HOW THE AI SYSTEM WORKS
# =========================================================

st.subheader("⚙️ How the AI Decision-Support System Works")


step1, step2, step3, step4 = st.columns(4)


step1.markdown(
    """
    ### 1️⃣
    **Data Collection**

    Public health, socioeconomic,
    demographic and geographic
    indicators are collected.
    """
)


step2.markdown(
    """
    ### 2️⃣
    **AI Analysis**

    The Machine Learning model
    analyzes the indicators and
    identifies vulnerability patterns.
    """
)


step3.markdown(
    """
    ### 3️⃣
    **Risk Prediction**

    Counties are classified as
    High, Medium or Low
    vulnerability.
    """
)


step4.markdown(
    """
    ### 4️⃣
    **Decision Support**

    Results support healthcare
    planning, prioritization and
    resource allocation.
    """
)


st.divider()


# =========================================================
# VULNERABILITY MAP PREVIEW
# =========================================================

st.subheader("🗺️ Kenya Vulnerability Map")


st.write(
    "Quick preview of county-level vulnerability. "
    "Open **AI Vulnerability Map** from the sidebar for the full interactive map."
)


kenya_map = folium.Map(
    location=[
        -0.0236,
        37.9062
    ],
    zoom_start=6,
    tiles="OpenStreetMap"
)


for _, row in df.iterrows():

    vulnerability = row[
        "Vulnerability_Level"
    ]

    if vulnerability == "High":

        color = "red"

    elif vulnerability == "Medium":

        color = "orange"

    else:

        color = "green"


    folium.CircleMarker(

        location=[
            row["Latitude"],
            row["Longitude"]
        ],

        radius=7,

        tooltip=(
            f"{row['County']} — "
            f"{vulnerability}"
        ),

        popup=f"""
        <b>County:</b> {row['County']}<br>
        <b>Vulnerability:</b> {vulnerability}<br>
        <b>Score:</b> {row['Vulnerability_Score']:.2f}
        """,

        color=color,

        fill=True,

        fill_color=color,

        fill_opacity=0.8

    ).add_to(kenya_map)


# =========================================================
# MAP LEGEND
# =========================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 220px;
z-index: 9999;
background-color: white;
border: 2px solid grey;
border-radius: 8px;
padding: 12px;
font-size: 14px;
box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">

<b>🗺️ Vulnerability Legend</b>
<br><br>

<span style="color:red;">●</span>
<b>High</b> — Priority intervention
<br>

<span style="color:orange;">●</span>
<b>Medium</b> — Monitor and improve
<br>

<span style="color:green;">●</span>
<b>Low</b> — Maintain and monitor

</div>
"""


kenya_map.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


st_folium(
    kenya_map,
    width=None,
    height=600,
    use_container_width=True
)


st.divider()


# =========================================================
# VULNERABILITY SUMMARY
# =========================================================

st.subheader("📊 Vulnerability Situation")


high_count = (
    df["Vulnerability_Level"]
    == "High"
).sum()


medium_count = (
    df["Vulnerability_Level"]
    == "Medium"
).sum()


low_count = (
    df["Vulnerability_Level"]
    == "Low"
).sum()


col1, col2, col3 = st.columns(3)


col1.metric(
    "🔴 High Vulnerability",
    high_count
)


col2.metric(
    "🟠 Medium Vulnerability",
    medium_count
)


col3.metric(
    "🟢 Low Vulnerability",
    low_count
)


st.divider()


# =========================================================
# HIGHEST-RISK COUNTY
# =========================================================

highest_score = df[
    "Vulnerability_Score"
].max()


highest_county = df.loc[
    df["Vulnerability_Score"].idxmax(),
    "County"
]


lowest_score = df[
    "Vulnerability_Score"
].min()


lowest_county = df.loc[
    df["Vulnerability_Score"].idxmin(),
    "County"
]


st.subheader("🚨 Public Health Risk Highlights")


col1, col2 = st.columns(2)


with col1:

    st.error(
        f"""
        ### 🔴 Highest-Risk County

        **{highest_county}**

        Vulnerability Score:
        **{highest_score:.2f}**
        """
    )


with col2:

    st.success(
        f"""
        ### 🟢 Lowest-Risk County

        **{lowest_county}**

        Vulnerability Score:
        **{lowest_score:.2f}**
        """
    )


st.divider()


# =========================================================
# NAVIGATION GUIDE
# =========================================================

st.subheader("🧭 Explore the AI System")


st.markdown(
    """
    Use the **sidebar navigation** to explore the complete system:

    **🤖 AI Prediction**  
    Predict vulnerability for a county using the trained Machine Learning model.

    **🗺️ AI Vulnerability Map**  
    Explore the geographic distribution of vulnerability across Kenya.

    **🚨 Early Warning System**  
    Identify high-risk counties and priority intervention areas.

    **🧠 Explainable AI**  
    Understand which public health indicators influence the AI model.

    **📊 Analytics Dashboard**  
    Explore statistical relationships and public health patterns.

    **🔮 AI Scenario Simulation**  
    Test "what-if" intervention scenarios using the AI model.

    **📈 AI Model Performance**  
    Examine model accuracy, confusion matrix and classification performance.
    """
)


st.divider()


# =========================================================
# PROJECT OBJECTIVES
# =========================================================

st.subheader("🎯 Project Objectives")


objective1, objective2 = st.columns(2)


with objective1:

    st.markdown(
        """
        - Identify vulnerable counties
        - Predict public health vulnerability
        - Detect priority intervention areas
        - Support healthcare resource allocation
        """
    )


with objective2:

    st.markdown(
        """
        - Improve evidence-based decision making
        - Provide explainable AI insights
        - Simulate possible interventions
        - Support public health planning
        """
    )


st.divider()


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.subheader("🏥 About the Project")


st.markdown(
    """
    **AI-Driven Public Health Vulnerability Index and Decision
    Support System for Kenyan Counties** is an AI and data science
    project designed to support public health decision making.

    The system combines:

    - Machine Learning
    - Data Analytics
    - Explainable AI
    - Geographic Visualization
    - Early Warning
    - Scenario Simulation
    - Decision Support
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.markdown(
    """
    ### 🏥 AI-Driven Public Health Vulnerability Index

    **Developed by Joy Kaaria**  
    BSc Information Technology, Karatina University  
    **ENGAGE Project 2026**

    *AI-assisted public health decision support for Kenyan counties.*
    """
)
# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
