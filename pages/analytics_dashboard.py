import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Public Health Analytics Dashboard",
    page_icon="📊",
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

st.title("📊 Public Health Analytics Dashboard")

st.markdown("""
Explore public health vulnerability patterns across Kenyan counties
using interactive data visualizations and statistical analysis.
""")

st.divider()


# ==============================
# KEY PERFORMANCE INDICATORS
# ==============================

st.subheader("🇰🇪 National Public Health Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Counties",
    len(df)
)

col2.metric(
    "Average Vulnerability Score",
    f"{df['Vulnerability_Score'].mean():.2f}"
)

col3.metric(
    "Average Poverty Rate",
    f"{df['Poverty_Rate'].mean():.1f}%"
)

col4.metric(
    "Average Immunization",
    f"{df['Immunization_Coverage'].mean():.1f}%"
)

st.divider()


# ==============================
# VULNERABILITY DISTRIBUTION
# ==============================

st.subheader("📈 Vulnerability Distribution")

col1, col2 = st.columns(2)

with col1:

    vulnerability_counts = (
        df["Vulnerability_Level"]
        .value_counts()
        .reset_index()
    )

    vulnerability_counts.columns = [
        "Vulnerability_Level",
        "Count"
    ]

    fig = px.bar(
        vulnerability_counts,
        x="Vulnerability_Level",
        y="Count",
        color="Vulnerability_Level",
        title="Number of Counties by Vulnerability Level"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.pie(
        df,
        names="Vulnerability_Level",
        title="Vulnerability Composition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ==============================
# TOP 10 VULNERABLE COUNTIES
# ==============================

st.subheader("🚨 Top 10 Most Vulnerable Counties")

top10 = df.sort_values(
    "Vulnerability_Score",
    ascending=False
).head(10)

fig = px.bar(
    top10,
    x="County",
    y="Vulnerability_Score",
    color="Vulnerability_Level",
    title="Top 10 Counties by Vulnerability Score",
    hover_data=[
        "Poverty_Rate",
        "Immunization_Coverage",
        "Health_Facilities"
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==============================
# POVERTY VS VULNERABILITY
# ==============================

st.subheader("📉 Poverty Rate vs Vulnerability Score")

fig = px.scatter(
    df,
    x="Poverty_Rate",
    y="Vulnerability_Score",
    color="Vulnerability_Level",
    hover_name="County",
    title="Relationship Between Poverty and Vulnerability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    """
This visualization helps identify whether counties with higher
poverty rates also tend to have higher vulnerability scores.

The chart shows association, not necessarily causation.
"""
)

st.divider()


# ==============================
# IMMUNIZATION VS VULNERABILITY
# ==============================

st.subheader(
    "💉 Immunization Coverage vs Vulnerability Score"
)

fig = px.scatter(
    df,
    x="Immunization_Coverage",
    y="Vulnerability_Score",
    color="Vulnerability_Level",
    hover_name="County",
    title="Immunization Coverage and Public Health Vulnerability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==============================
# HEALTH FACILITIES ANALYSIS
# ==============================

st.subheader(
    "🏥 Healthcare Facilities by Vulnerability Level"
)

fig = px.box(
    df,
    x="Vulnerability_Level",
    y="Health_Facilities",
    color="Vulnerability_Level",
    title="Healthcare Facilities Across Vulnerability Groups"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==============================
# CORRELATION ANALYSIS
# ==============================

st.subheader("🔗 Public Health Feature Correlation")

st.write(
    "The correlation matrix shows statistical relationships "
    "between numerical public health indicators."
)

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

corr = numeric_df.corr()

fig, ax = plt.subplots(
    figsize=(14, 10)
)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

st.caption(
    "Correlation indicates statistical association and does not prove causation."
)

st.divider()


# ==============================
# COUNTY COMPARISON
# ==============================

st.subheader("🏥 Interactive County Comparison")

selected_counties = st.multiselect(
    "Select counties to compare",
    options=df["County"].unique(),
    default=df["County"].unique()[:3]
)

if selected_counties:

    comparison_df = df[
        df["County"].isin(selected_counties)
    ]

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    fig = px.bar(
        comparison_df,
        x="County",
        y="Vulnerability_Score",
        color="Vulnerability_Level",
        title="Selected County Vulnerability Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.warning(
        "Please select at least one county."
    )


# ==============================
# DATA DOWNLOAD
# ==============================

st.divider()

st.subheader("📥 Download Analytics Data")

csv = df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Public Health Dataset",
    data=csv,
    file_name="kenya_public_health_analytics.csv",
    mime="text/csv"
)


# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "AI-Driven Public Health Vulnerability Index and "
    "Decision Support System for Kenyan Counties | ENGAGE Project 2026"
)