# 🏥 AI-Driven Public Health Vulnerability Index and Decision Support System for Kenyan Counties

An AI-powered decision-support platform that uses Machine Learning, data analytics, geospatial visualization, Explainable AI, and scenario simulation to identify public health vulnerability across Kenyan counties.

**Developed by Joy Kaaria**  
Bachelor of Science in Information Technology  
Karatina University  

**ENGAGE Project 2026**

---

# 🚀 Live Demo

## 🔗 Project Links

- 🚀 Live Streamlit Application: https://ai-public-health-vulnerability.streamlit.app
- 💻 GitHub Repository: https://github.com/Gaicug/AI_Public_Health_Project

---

# 🌍 Project Overview

Public health resources are often limited and must be strategically allocated to areas with the greatest need.

This project applies **Artificial Intelligence and Machine Learning** to analyze public health, healthcare, socioeconomic, and geographic indicators across Kenyan counties.

The system supports:

- Public health vulnerability prediction
- Geographic vulnerability visualization
- Early warning and risk prioritization
- Explainable AI
- Public health analytics
- AI scenario simulation
- Machine Learning model evaluation
- Evidence-based decision support

---

# 🎯 Project Objectives

The project aims to:

1. Identify vulnerable counties using data-driven methods.
2. Apply Machine Learning to predict vulnerability levels.
3. Support evidence-based healthcare resource allocation.
4. Visualize geographic patterns of vulnerability.
5. Provide early warning for high-risk counties.
6. Improve transparency through Explainable AI.
7. Support intervention planning using AI scenario simulation.
8. Evaluate and communicate Machine Learning model performance.

---

# 🤖 Artificial Intelligence Component

The system uses a supervised Machine Learning model trained using county-level public health indicators.

The AI model analyzes indicators such as:

- Population
- Population Density
- Poverty Rate
- Health Facilities
- Doctors per 10,000 people
- Immunization Coverage
- Clean Water Access
- Maternal Mortality
- Under-5 Mortality
- Malnutrition Rate
- HIV Prevalence
- Malaria Prevalence
- Unemployment Rate
- Geographic Location
- Vulnerability Score

The model predicts one of three vulnerability categories:

| Level | Meaning |
|---|---|
| 🔴 High | Priority intervention required |
| 🟠 Medium | Increased monitoring and preventive intervention |
| 🟢 Low | Continue monitoring and maintain interventions |

The trained model is stored in:

```text
models/vulnerability_model.pkl