import streamlit as st
import pandas as pd
import plotly.express as px

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Early Warning System",
    page_icon="🚨",
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

df = load_data()

# ==============================
# PAGE TITLE
# ==============================

st.title("🚨 Public Health Early Warning System")

st.markdown("""
This page identifies counties that may require priority public health
intervention based on their vulnerability level and vulnerability score.

The system supports early identification of high-risk areas to assist
with healthcare planning and resource allocation.
""")

st.divider()

# ==============================
# CALCULATE RISK COUNTS
# ==============================

high_risk = df[
    df["Vulnerability_Level"] == "High"
]

medium_risk = df[
    df["Vulnerability_Level"] == "Medium"
]

low_risk = df[
    df["Vulnerability_Level"] == "Low"
]

# ==============================
# EARLY WARNING KPIs
# ==============================

st.subheader("🚨 National Early Warning Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🚨 High-Risk Counties",
    len(high_risk)
)

col2.metric(
    "⚠️ Medium-Risk Counties",
    len(medium_risk)
)

col3.metric(
    "✅ Low-Risk Counties",
    len(low_risk)
)

col4.metric(
    "Highest Risk Score",
    f"{df['Vulnerability_Score'].max():.2f}"
)

st.divider()

# ==============================
# HIGH PRIORITY ALERT
# ==============================

st.subheader("🚨 High Priority Public Health Alerts")

if len(high_risk) > 0:

    st.error(
        f"""
        🚨 ALERT: {len(high_risk)} counties are currently classified
        as HIGH VULNERABILITY.

        These counties should be prioritized for public health
        assessment, intervention planning and resource allocation.
        """
    )

    alert_df = high_risk[
        [
            "County",
            "Vulnerability_Level",
            "Vulnerability_Score",
            "Poverty_Rate",
            "Immunization_Coverage",
            "Health_Facilities"
        ]
    ].sort_values(
        "Vulnerability_Score",
        ascending=False
    )

    st.dataframe(
        alert_df,
        use_container_width=True
    )

else:

    st.success(
        "No counties are currently classified as High Vulnerability."
    )

st.divider()

# ==============================
# TOP 10 PRIORITY COUNTIES
# ==============================

st.subheader("🏥 Top 10 Priority Counties")

top10 = df.sort_values(
    "Vulnerability_Score",
    ascending=False
).head(10)

priority_table = top10[
    [
        "County",
        "Vulnerability_Level",
        "Vulnerability_Score",
        "Poverty_Rate",
        "Immunization_Coverage",
        "Under5_Mortality"
    ]
]

priority_table = priority_table.copy()

priority_table.insert(
    0,
    "Priority Rank",
    range(1, len(priority_table) + 1)
)

st.dataframe(
    priority_table,
    use_container_width=True
)

# ==============================
# PRIORITY RANKING CHART
# ==============================

st.divider()

st.subheader("📊 Vulnerability Priority Ranking")

fig = px.bar(
    top10,
    x="County",
    y="Vulnerability_Score",
    color="Vulnerability_Level",
    title="Top 10 Counties Requiring Priority Attention",
    hover_data=[
        "Poverty_Rate",
        "Immunization_Coverage",
        "Under5_Mortality"
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==============================
# AUTOMATED DECISION SUPPORT
# ==============================

st.subheader("🧠 Automated Decision Support")

highest_county = df.loc[
    df["Vulnerability_Score"].idxmax(),
    "County"
]

highest_score = df[
    "Vulnerability_Score"
].max()

lowest_county = df.loc[
    df["Vulnerability_Score"].idxmin(),
    "County"
]

average_score = df[
    "Vulnerability_Score"
].mean()

st.info(
    f"""
    🤖 **AI-Assisted Public Health Summary**

    🚨 Highest priority county: **{highest_county}**

    📈 Highest vulnerability score: **{highest_score:.2f}**

    📊 National average vulnerability score: **{average_score:.2f}**

    🟢 Lowest vulnerability county: **{lowest_county}**

    The decision-support system recommends prioritizing counties
    with high vulnerability scores for healthcare interventions,
    disease prevention and resource allocation.
    """
)

st.divider()

# ==============================
# INTERVENTION PRIORITIES
# ==============================

st.subheader("🎯 Recommended Intervention Priorities")

st.markdown("""
### 🚨 High Vulnerability

- Increase healthcare funding.
- Deploy additional healthcare workers.
- Improve access to healthcare facilities.
- Strengthen immunization programmes.
- Improve water and sanitation services.
- Increase disease surveillance.

### ⚠️ Medium Vulnerability

- Strengthen preventive healthcare.
- Improve disease monitoring.
- Expand community health programmes.
- Monitor vulnerable populations.

### 🟢 Low Vulnerability

- Maintain existing public health interventions.
- Continue monitoring health indicators.
- Strengthen preparedness for emerging risks.
""")

st.divider()

# ==============================
# DOWNLOAD REPORT
# ==============================

st.subheader("📥 Download Priority County Report")

download_df = priority_table.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Early Warning Report",
    data=download_df,
    file_name="kenya_public_health_early_warning_report.csv",
    mime="text/csv"
)

st.caption(
    "This early warning system is designed as a public health "
    "decision-support and research tool."
)