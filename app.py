import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="AI Public Health Vulnerability Index",
    page_icon="🏥",
    layout="wide"
)


st.markdown(
"""
<style>
footer {visibility: hidden;}
</style>
""",
unsafe_allow_html=True
)




# Load data
df = pd.read_csv("data/kenya_public_health_vulnerability.csv")

# Load model
model = joblib.load("models/vulnerability_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
print(model.feature_names_in_)


# Title
st.title(
    "🏥 AI-Driven Public Health Vulnerability Index and Decision Support System for Kenyan Counties"
)
st.markdown("""
### 🇰🇪 Public Health Intelligence Dashboard

This AI-powered platform identifies vulnerable counties and supports evidence-based decision making for healthcare resource allocation in Kenya.
""")



st.subheader("Kenya Public Health Vulnerability Map")

kenya_map = folium.Map(
    location=[-0.0236, 37.9062],
    zoom_start=6
)

for _, row in df.iterrows():

    color = "green"

    if row["Vulnerability_Level"] == "High":
        color = "red"

    elif row["Vulnerability_Level"] == "Medium":
        color = "orange"

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=8,
        popup=f"""
        County: {row['County']}<br>
        Vulnerability: {row['Vulnerability_Level']}<br>
        Score: {row['Vulnerability_Score']}
        """,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8
    ).add_to(kenya_map)

    legend_html = """
<div style="
position: fixed;
bottom: 50px;
right: 50px;
width: 180px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding:10px;
border-radius:8px;
box-shadow:2px 2px 6px rgba(0,0,0,0.3);
">

<b>Vulnerability Level</b><br><br>

<span style="color:red;">&#9679;</span> High<br>

<span style="color:orange;">&#9679;</span> Medium<br>

<span style="color:green;">&#9679;</span> Low

</div>
"""

kenya_map.get_root().html.add_child(folium.Element(legend_html))

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 210px;
z-index: 9999;
background-color: white;
border: 2px solid grey;
border-radius: 8px;
padding: 12px;
font-size: 14px;
box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>🗺️ Vulnerability Legend</b><br><br>

<span style="color:red;">●</span>
<b>High</b> — Priority intervention<br>

<span style="color:orange;">●</span>
<b>Medium</b> — Monitor and improve<br>

<span style="color:green;">●</span>
<b>Low</b> — Maintain and monitor
</div>
"""

kenya_map.get_root().html.add_child(
    folium.Element(legend_html)
)

st_folium(
    kenya_map,
    width=900,
    height=600
)
st.subheader("County Vulnerability Table")

display_df = df[
    [
        "County",
        "Vulnerability_Level",
        "Vulnerability_Score",
        "Poverty_Rate",
        "Immunization_Coverage"
    ]
]

st.dataframe(
    display_df,
    use_container_width=True
)
csv = display_df.to_csv(index=False)

st.download_button(
    label="📥 Download Vulnerability Report",
    data=csv,
    file_name="kenya_vulnerability_report.csv",
    mime="text/csv"
)
st.markdown(
"""
This system identifies vulnerable counties using Artificial Intelligence
and supports evidence-based public health decision making.
"""
)

st.info("""
### 🎯 Objectives

✅ Identify vulnerable counties

✅ Predict public health risks using AI

✅ Support healthcare decision making

✅ Prioritize intervention allocation
""")
st.subheader("County Explorer")


selected_county = st.selectbox(
    "Select County",
    df["County"].unique()
)

county_data = df[
    df["County"] == selected_county
]

st.dataframe(county_data,
             use_container_width=True
           )
if not county_data.empty:

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Poverty Rate",
        f"{county_data['Poverty_Rate'].iloc[0]:.1f}%"
    )

    c2.metric(
        "Immunization",
        f"{county_data['Immunization_Coverage'].iloc[0]:.1f}%"
    )

    c3.metric(
        "Vulnerability Score",
        round(
            county_data['Vulnerability_Score'].iloc[0],
            2
        )
    )



# ======================
# Prediction Sidebar
# ======================

# ======================
# Prediction Sidebar
# ======================

st.sidebar.markdown(
"""
# 🏥 County Predictor

Enter county health indicators below
to predict vulnerability levels using AI.
"""
)

population = st.sidebar.number_input(
    "Population",
    value=int(df["Population"].mean())
)

population_density = st.sidebar.number_input(
    "Population Density",
    value=float(df["Population_Density"].mean())
)

poverty_rate = st.sidebar.number_input(
    "Poverty Rate",
    value=float(df["Poverty_Rate"].mean())
)

health_facilities = st.sidebar.number_input(
    "Health Facilities",
    value=int(df["Health_Facilities"].mean())
)

doctors = st.sidebar.number_input(
    "Doctors per 10000",
    value=float(df["Doctors_per_10000"].mean())
)



immunization = st.sidebar.number_input(
    "Immunization Coverage",
    value=float(df["Immunization_Coverage"].mean())
)
water = st.sidebar.number_input(
    "Clean Water Access",
    value=float(df["Clean_Water_Access"].mean())
)

maternal = st.sidebar.number_input(
    "Maternal Mortality",
    value=float(df["Maternal_Mortality"].mean())
)

under5 = st.sidebar.number_input(
    "Under5 Mortality",
    value=float(df["Under5_Mortality"].mean())
)

malnutrition = st.sidebar.number_input(
    "Malnutrition Rate",
    value=float(df["Malnutrition_Rate"].mean())
)

hiv = st.sidebar.number_input(
    "HIV Prevalence",
    value=float(df["HIV_Prevalence"].mean())
)

malaria = st.sidebar.number_input(
    "Malaria Prevalence",
    value=float(df["Malaria_Prevalence"].mean())
)

unemployment = st.sidebar.number_input(
    "Unemployment Rate",
    value=float(df["Unemployment_Rate"].mean())
)
latitude = st.sidebar.number_input(
    "Latitude",
    value=float(df["Latitude"].mean())
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=float(df["Longitude"].mean())
)

vulnerability_score = st.sidebar.number_input(
    "Vulnerability Score",
    value=float(df["Vulnerability_Score"].mean())
)


if st.sidebar.button("Predict Vulnerability"):

    
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
    "Latitude": [latitude],
    "Longitude": [longitude],
    "Vulnerability_Score": [vulnerability_score]

})
    input_data = input_data[model.feature_names_in_]
  

    with st.spinner("Running AI model..."):
        prediction = model.predict(input_data)

    result = encoder.inverse_transform(prediction)

    st.sidebar.success(
        f"Predicted Vulnerability: {result[0]}"
    )

    # Recommendations
    if result[0] == "High":

        st.error("""
        🔴 High Vulnerability

        Recommendations:
        • Increase healthcare funding
        • Deploy mobile clinics
        • Improve sanitation
        • Strengthen disease surveillance
        • Increase immunization campaigns
        """)

    elif result[0] == "Medium":

        st.warning("""
        🟡 Medium Vulnerability

        Recommendations:
        • Improve preventive healthcare
        • Expand immunization programmes
        • Increase disease monitoring
        """)

    else:

        st.success("""
        🟢 Low Vulnerability

        Recommendations:
        • Maintain interventions
        • Continue monitoring
        • Sustain public health funding
        """)

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Counties",
    len(df)
)

col2.metric(
    "Average Poverty",
    f"{df['Poverty_Rate'].mean():.1f}%"
)

col3.metric(
    "Average Immunization",
    f"{df['Immunization_Coverage'].mean():.1f}%"
)

col4.metric(
    "Highest Score",
    round(
        df["Vulnerability_Score"].max(),
        2
    )
)

# Dataset
st.subheader("County Dataset")

st.dataframe(df)

# Distribution
st.subheader("Vulnerability Distribution")
st.subheader("County Vulnerability Composition")

fig3 = px.pie(
    df,
    names="Vulnerability_Level",
    title="Distribution of Vulnerability Levels"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)
st.subheader("Feature Correlation")

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

corr = numeric_df.corr()

fig, ax = plt.subplots(
    figsize=(12,8)
)
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

fig = px.histogram(
    df,
    x="Vulnerability_Level",
    color="Vulnerability_Level"
)

st.plotly_chart(fig, use_container_width=True)

# Top vulnerable counties
st.subheader("County Vulnerability Ranking")

top10 = df.sort_values(
    "Vulnerability_Score",
    ascending=False
).head(10)

fig2 = px.bar(
    top10,
    x="County",
    y="Vulnerability_Score",
    color="Vulnerability_Level",
    title="Top 10 Most Vulnerable Counties"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

top = df.sort_values(
    "Vulnerability_Score",
    ascending=False
)

fig2 = px.bar(
    top,
    x="County",
    y="Vulnerability_Score",
    color="Vulnerability_Level"
)

st.subheader("🤖 AI Insights")

highest = df.loc[
    df["Vulnerability_Score"].idxmax(),
    "County"
]

lowest = df.loc[
    df["Vulnerability_Score"].idxmin(),
    "County"
]
st.subheader("📈 National Vulnerability Indicator")

average_score = df["Vulnerability_Score"].mean()

fig_gauge = px.scatter(
    x=[0],
    y=[average_score]
)

st.metric(
    "Average National Vulnerability Score",
    round(average_score, 2)
)

st.info(
    f"""
    Most vulnerable county: {highest}

    Least vulnerable county: {lowest}

    The AI model suggests prioritizing
    healthcare interventions in highly
    vulnerable counties.
    """
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.markdown(
    """
    ## 🏥 AI-Driven Public Health Vulnerability Index

    ### Decision Support System for Kenyan Counties

    Developed by **Joy Kaaria**  
    Bachelor of Science in Information Technology  
    Karatina University  

    **ENGAGE Project 2026**
    """
)