# 🏥 AI-Driven Public Health Vulnerability Index and Decision Support System for Kenyan Counties

An AI-powered decision-support platform that analyzes public health and socioeconomic indicators to identify vulnerable Kenyan counties and support evidence-based healthcare resource allocation.

**Developed by Joy Kaaria**
Bachelor of Science in Information Technology
Karatina University
**ENGAGE Project 2026**

---

## 🌍 Project Overview

Public health resources are often limited and must be strategically allocated to areas with the greatest need.

This project uses **machine learning, data analytics, and geospatial visualization** to assess public health vulnerability across Kenyan counties.

The system analyzes indicators including:

* Population
* Population density
* Poverty rate
* Health facilities
* Doctors per 10,000 people
* Immunization coverage
* Clean water access
* Maternal mortality
* Under-five mortality
* Malnutrition
* HIV prevalence
* Malaria prevalence
* Unemployment
* Vulnerability score

The resulting platform provides an interactive dashboard that helps users identify high-risk counties and explore potential intervention priorities.

---

## 🎯 Objectives

The project aims to:

1. Identify counties experiencing higher levels of public health vulnerability.
2. Apply machine learning to classify vulnerability levels.
3. Visualize vulnerability geographically across Kenya.
4. Provide county-level public health indicators.
5. Support evidence-based healthcare resource allocation.
6. Provide an interactive prediction interface.
7. Generate data-driven insights for decision makers.

---

## 🤖 Artificial Intelligence Component

The system uses a supervised machine learning model trained using county-level public health indicators.

The model predicts one of three vulnerability categories:

| Level     | Meaning                                          |
| --------- | ------------------------------------------------ |
| 🔴 High   | Priority intervention required                   |
| 🟠 Medium | Increased monitoring and preventive intervention |
| 🟢 Low    | Continue monitoring and maintain interventions   |

The trained model is stored in:

```text
models/vulnerability_model.pkl
```

The label encoder is stored in:

```text
models/label_encoder.pkl
```

---

## 🗺️ Interactive Kenya Vulnerability Map

The dashboard includes an interactive geographical visualization of Kenyan counties.

Each county is represented using a color-coded marker:

* 🔴 **Red — High Vulnerability**
* 🟠 **Orange — Medium Vulnerability**
* 🟢 **Green — Low Vulnerability**

Users can interact with the map to explore county-level vulnerability information.

---

## 📊 Dashboard Features

### 1. Public Health Vulnerability Map

Interactive map displaying county vulnerability levels across Kenya.

### 2. County Explorer

Users can select an individual county and examine its public health indicators.

### 3. AI County Predictor

Users can enter public health indicators and receive an AI-generated vulnerability classification.

### 4. Key Performance Indicators

The dashboard displays:

* Total number of counties
* Average poverty rate
* Average immunization coverage
* Highest vulnerability score
* National average vulnerability score

### 5. Vulnerability Distribution

Interactive charts show the distribution of counties across:

* High
* Medium
* Low vulnerability

### 6. County Vulnerability Ranking

The system ranks counties according to their vulnerability scores and highlights the most vulnerable counties.

### 7. Feature Correlation Analysis

A correlation heatmap provides insight into relationships between numerical public health indicators.

### 8. AI-Generated Insights

The system automatically identifies:

* Highest-risk county
* Lowest-risk county
* National average vulnerability score
* Number of high-, medium-, and low-vulnerability counties

### 9. Vulnerability Report

Users can download a CSV report containing key county vulnerability information.

---

## 🧠 System Architecture

```text
                ┌─────────────────────────┐
                │     County Dataset      │
                │ Public Health Indicators│
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    Data Preparation     │
                │ Cleaning & Processing   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    Machine Learning     │
                │   Vulnerability Model   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │     Prediction Engine   │
                │ High / Medium / Low     │
                └────────────┬────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │       Streamlit Decision Dashboard     │
        ├────────────────────────────────────────┤
        │ • Kenya Vulnerability Map              │
        │ • County Explorer                      │
        │ • AI Predictor                         │
        │ • KPIs                                 │
        │ • Charts & Analytics                   │
        │ • AI Insights                          │
        │ • Downloadable Reports                 │
        └────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Joblib

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Geospatial Visualization

* Folium
* Streamlit-Folium

### Web Application

* Streamlit

### Development Tools

* Visual Studio Code
* Jupyter Notebook
* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

## 📁 Project Structure

```text
AI_Public_Health_Project/
│
├── app.py
│
├── train_model.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   └── kenya_public_health_vulnerability.csv
│
├── models/
│   ├── vulnerability_model.pkl
│   └── label_encoder.pkl
│
└── notebooks/
    ├── EDA.ipynb
    └── model_training.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Gaicug/AI_Public_Health_Project.git
```

Navigate into the project:

```bash
cd AI_Public_Health_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 📈 Model Training

The model-training workflow is contained in:

```text
train_model.py
```

Exploratory data analysis and model development notebooks are available in:

```text
notebooks/
```

---

## ☁️ Live Application

The project is deployed using **Streamlit Community Cloud**.

**Live Demo:**
Add your Streamlit Cloud URL here.

```text
https://your-app-name.streamlit.app
```

---

## 🔗 Repository

GitHub repository:

```text
https://github.com/Gaicug/AI_Public_Health_Project
```

---

## 💡 Potential Impact

The system demonstrates how artificial intelligence and data analytics can be applied to public health planning.

Potential applications include:

* Healthcare resource prioritization
* County-level risk monitoring
* Public health planning
* Disease prevention planning
* Identification of vulnerable populations
* Data-driven intervention planning
* Visualization of regional health disparities

The platform is designed as a **decision-support tool**, rather than a replacement for public health professionals or government decision-making processes.

---

## 🚀 Future Improvements

Future versions could incorporate:

* Real-time public health datasets
* Additional socioeconomic indicators
* Time-series vulnerability forecasting
* Explainable AI techniques
* Automated intervention recommendations
* Integration with government and health information systems
* More advanced geospatial analysis
* County-level historical trend analysis
* Model performance monitoring
* Mobile-friendly optimization

---

## 👩‍💻 Developer

**Joy Kaaria**

Bachelor of Science in Information Technology
Karatina University

**ENGAGE Project 2026**

---

## 📜 Disclaimer

This project is an academic and research-oriented decision-support prototype.

The vulnerability classifications and recommendations should not be interpreted as official government health assessments or clinical advice.

---

## ⭐ Acknowledgement

This project demonstrates the application of:

**Artificial Intelligence + Data Analytics + Geospatial Visualization + Decision Support**

to a real-world public health challenge in Kenya.
