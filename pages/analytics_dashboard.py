import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


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
# DESCRIPTIVE STATISTICS
# ==============================

st.subheader("📊 Descriptive Statistics")

st.write(
    """
    Descriptive statistics summarize the distribution of important
    public health indicators across the 47 Kenyan counties.
    """
)

statistical_features = [
    "Population",
    "Population_Density",
    "Poverty_Rate",
    "Health_Facilities",
    "Doctors_per_10000",
    "Immunization_Coverage",
    "Maternal_Mortality",
    "Under5_Mortality",
    "Malnutrition_Rate",
    "HIV_Prevalence",
    "Malaria_Prevalence",
    "Unemployment_Rate",
    "Vulnerability_Score"
]

descriptive_stats = df[statistical_features].describe().T

descriptive_stats = descriptive_stats[
    ["count", "mean", "std", "min", "50%", "max"]
]

descriptive_stats.columns = [
    "Count",
    "Mean",
    "Std. Deviation",
    "Minimum",
    "Median",
    "Maximum"
]

st.dataframe(
    descriptive_stats.round(2),
    use_container_width=True
)

st.info(
    """
    **Interpretation:** The mean represents the average value,
    the median represents the middle observation, and the standard
    deviation measures how widely county values vary around the mean.
    """
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
# STATISTICAL SIGNIFICANCE
# ==============================

st.subheader("🧪 Statistical Significance Analysis")

st.write(
    """
    Pearson correlation is used to measure the strength and direction
    of the linear relationship between public health indicators and
    the Vulnerability Score.
    """
)

correlation_features = [
    "Poverty_Rate",
    "Population_Density",
    "Health_Facilities",
    "Doctors_per_10000",
    "Immunization_Coverage",
    "Maternal_Mortality",
    "Under5_Mortality",
    "Malnutrition_Rate",
    "HIV_Prevalence",
    "Malaria_Prevalence",
    "Unemployment_Rate"
]

results = []

for feature in correlation_features:

    r, p = pearsonr(
        df[feature],
        df["Vulnerability_Score"]
    )

    if p < 0.05:
        significance = "Statistically Significant"
    else:
        significance = "Not Statistically Significant"

    results.append({
        "Indicator": feature,
        "Correlation (r)": r,
        "P-value": p,
        "Significance": significance
    })

significance_df = pd.DataFrame(results)

significance_df = significance_df.sort_values(
    "Correlation (r)",
    ascending=False
)

st.dataframe(
    significance_df.style.format({
        "Correlation (r)": "{:.3f}",
        "P-value": "{:.4f}"
    }),
    use_container_width=True
)

st.caption(
    "A p-value below 0.05 is commonly interpreted as statistically significant. "
    "Statistical significance does not establish causation."
)

st.divider()

# ==============================
# REGRESSION ANALYSIS
# ==============================

st.subheader("📐 Regression Analysis")

st.write(
    """
    A multiple linear regression model estimates how public health
    indicators are associated with the Vulnerability Score when
    considered together.
    """
)

regression_features = [
    "Poverty_Rate",
    "Doctors_per_10000",
    "Immunization_Coverage",
    "Maternal_Mortality",
    "Under5_Mortality",
    "Malnutrition_Rate",
    "Malaria_Prevalence",
    "Unemployment_Rate"
]

X = df[regression_features]
y = df["Vulnerability_Score"]

regression_model = LinearRegression()

regression_model.fit(X, y)

predictions = regression_model.predict(X)

r_squared = r2_score(
    y,
    predictions
)

st.metric(
    "R² — Model Explanatory Power",
    f"{r_squared:.3f}"
)

regression_results = pd.DataFrame({
    "Indicator": regression_features,
    "Coefficient": regression_model.coef_
})

regression_results["Direction"] = regression_results[
    "Coefficient"
].apply(
    lambda x: "Positive association"
    if x > 0
    else "Negative association"
)

regression_results = regression_results.sort_values(
    "Coefficient",
    ascending=False
)

st.dataframe(
    regression_results.style.format({
        "Coefficient": "{:.4f}"
    }),
    use_container_width=True
)
# ==============================
# ACTUAL VS PREDICTED VALUES
# ==============================

st.subheader("📊 Actual vs Predicted Vulnerability Score")

regression_plot = pd.DataFrame({
    "Actual Score": y,
    "Predicted Score": predictions
})

fig = px.scatter(
    regression_plot,
    x="Actual Score",
    y="Predicted Score",
    title="Actual vs Predicted Vulnerability Scores",
    labels={
        "Actual Score": "Actual Vulnerability Score",
        "Predicted Score": "Predicted Vulnerability Score"
}
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    """
    **R²** indicates the proportion of variation in Vulnerability Score
    explained by the predictors included in this regression model.
    Regression coefficients indicate the direction of the association
    while holding the other included predictors constant.
    """
)

st.divider()
# ==============================
# STATISTICAL INSIGHTS
# ==============================

st.subheader("🧠 Statistical Insights")

strongest_positive = significance_df.loc[
    significance_df["Correlation (r)"].idxmax()
]

strongest_negative = significance_df.loc[
    significance_df["Correlation (r)"].idxmin()
]

significant_features = significance_df[
    significance_df["P-value"] < 0.05
]

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Strongest Positive Association",
        strongest_positive["Indicator"]
    )

with col2:

    st.metric(
        "Strongest Negative Association",
        strongest_negative["Indicator"]
    )

with col3:

    st.metric(
        "Significant Indicators",
        len(significant_features)
    )

st.markdown("### 📌 Interpretation")

st.write(
    f"""
    **{strongest_positive['Indicator']}** has the strongest positive
    correlation with the Vulnerability Score among the indicators
    analyzed, with a correlation coefficient of
    **{strongest_positive['Correlation (r)']:.3f}**.

    **{strongest_negative['Indicator']}** has the strongest negative
    correlation, with a coefficient of
    **{strongest_negative['Correlation (r)']:.3f}**.

    A total of **{len(significant_features)} indicators** have
    p-values below 0.05 in this analysis.
    """
)

st.caption(
    "These findings describe statistical associations within this dataset "
    "and should not be interpreted as evidence of causal relationships."
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