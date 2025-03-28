import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load trained model
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# Set page configuration
st.set_page_config(page_title="🔍 Employee Attrition Prediction", layout="wide")

# Apply background image using CSS
def add_bg_from_local(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_file}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("D:/Employment attrition prediction analysis/istockphoto-927747938-612x612.jpg") 

# Load trained model
model = load_model('best_lgb_model.pkl')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏙️ Home","📊 Exploratory Data Analysis", "🔮 Predict Attrition"])

# File Upload
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Please upload a dataset to proceed.")
    st.stop()

# ---- Home Page ----
if page == "🏙️ Home":
    st.title("Employee Attrition Prediction")
    st.markdown("""
### 🚀 **Project Overview**  
This app helps HR teams analyze employee data and predict attrition risk based on key factors like job satisfaction, salary, and work experience.

### 🎯 **Project Goals**
✔ **Analyze** employee attrition trends  
✔ **Identify** key factors leading to attrition  
✔ **Predict** whether an employee is at risk of leaving  
✔ **Provide actionable insights** for HR decision-making  

## 🛠️ **How It Works**
### 1️⃣ **Data Collection**
We use employee datasets with features like:
- **Personal Factors**: Age, Gender, Marital Status  
- **Work-related Factors**: Job Role, Department, Salary, Job Satisfaction  
- **Performance Metrics**: Years at Company, Promotions, Work-Life Balance  

### 2️⃣ **Data Preprocessing**
- Handling missing values and encoding categorical variables  
- Normalizing numerical features for better model performance  

### 3️⃣ **Machine Learning Model**
We train and evaluate different models:
- **Logistic Regression**
- **decision Tree**
- **Random Forest**
- **XGBoost**
- **Light Gradient Boosting**

📌 **Best model is selected based on Accuracy, F1-Score, and ROC-AUC.**

### 4️⃣ **Predictions**
- Enter employee details  
- Predict whether the employee is likely to **Stay (0) or Leave (1)**  
- Show confidence score and insights  

---

## 📊 **Visual Insights**
🔹 Attrition trends by department and job roles  
🔹 Salary vs. attrition rate analysis  
🔹 Work-life balance impact on attrition  
🔹 Employee satisfaction levels  

""")



# ---- EDA Page ----
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head(5))

    # Summary statistics
    st.subheader("Dataset Summary")
    st.write(df.describe())

    # Attrition Distribution
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Attrition", data=df, palette="pastel", ax=ax)
    st.pyplot(fig)

    # Attrition by Monthly Income
    st.subheader("Attrition by Monthly Income")
    fig, ax = plt.subplots()
    sns.barplot(x="Attrition", y="MonthlyIncome", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Attrition by Job Role
    st.subheader("Attrition by Job Role")
    fig, ax = plt.subplots()
    sns.countplot(x="JobRole", hue="Attrition", data=df, palette="Set2", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# ---- Prediction Page ----
elif page == "🔮 Predict Attrition":
    st.title("🔮 Employee Attrition Prediction")

    # Define input fields
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=65, value=30)
        DailyRate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=500)
        DistanceFromHome = st.number_input("Distance From Home (km)", min_value=1, max_value=50, value=10)
        Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        HourlyRate = st.number_input("Hourly Rate", min_value=10, max_value=100, value=30)

    with col2:
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=3)
        PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=5, max_value=25, value=15)
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        WorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        YearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, max_value=20, value=4)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2)

    # Categorical features
    BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
    Department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    Gender = st.selectbox("Gender", ["Female", "Male"])
    OverTime = st.selectbox("OverTime", ["No", "Yes"])

    # One-hot encoding
    categorical_features = {
        "BusinessTravel_Non-Travel": 1 if BusinessTravel == "Non-Travel" else 0,
        "BusinessTravel_Travel_Frequently": 1 if BusinessTravel == "Travel_Frequently" else 0,
        "BusinessTravel_Travel_Rarely": 1 if BusinessTravel == "Travel_Rarely" else 0,
        "Department_Human Resources": 1 if Department == "Human Resources" else 0,
        "Department_Research & Development": 1 if Department == "Research & Development" else 0,
        "Department_Sales": 1 if Department == "Sales" else 0,
        "Gender_Female": 1 if Gender == "Female" else 0,
        "Gender_Male": 1 if Gender == "Male" else 0,
        "OverTime_No": 1 if OverTime == "No" else 0,
        "OverTime_Yes": 1 if OverTime == "Yes" else 0,
    }

    # Prepare input data
    input_data = {**categorical_features, "Age": Age, "DailyRate": DailyRate, "DistanceFromHome": DistanceFromHome,
                  "Education": Education, "TrainingTimesLastYear": TrainingTimesLastYear, "EnvironmentSatisfaction": EnvironmentSatisfaction,
                  "YearsAtCompany": YearsAtCompany, "HourlyRate": HourlyRate, "MonthlyIncome": MonthlyIncome, "NumCompaniesWorked": NumCompaniesWorked,
                  "PercentSalaryHike": PercentSalaryHike, "StockOptionLevel": StockOptionLevel, "WorkLifeBalance": WorkLifeBalance,
                  "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion}

    input_df = pd.DataFrame([input_data])

    # Ensure input data matches trained model features
    expected_features = model.booster_.feature_name()
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[expected_features]

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]
        if prediction[0] == 1:
            st.error(f"⚠️ The employee is at HIGH risk of leaving! (Attrition Probability: {probability[0]:.2f})")
        else:
            st.success(f"✅ The employee is likely to STAY! (Attrition Probability: {probability[0]:.2f})")