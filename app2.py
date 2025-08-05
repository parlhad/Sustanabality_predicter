import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout and page title
st.set_page_config(page_title="🌱 Sustainability Predictor", layout="wide")

# Load model
model = joblib.load("sustainability.pkl")

# Sidebar inputs – Real Features
st.sidebar.header("📥 Input Features for Prediction")

def user_input_features():
    energy = st.sidebar.number_input("🔋 Energy Usage (kWh)", min_value=0.0, max_value=1000.0, value=500.0)
    waste = st.sidebar.number_input("🗑️ Waste Generation (tons)", min_value=0.0, max_value=100.0, value=20.0)
    renewable = st.sidebar.slider("🌞 Renewable Usage (%)", 0, 100, 50)
    carbon = st.sidebar.slider("🌫️ Carbon Emission (kg)", 0, 1000, 300)

    data = {
        'Energy_Usage': energy,
        'Waste_Generation': waste,
        'Renewable_Usage': renewable,
        'Carbon_Emission': carbon
    }
    return pd.DataFrame([data])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("green_tech_data.csv")
    return df

df = load_data()

# Layout: Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visuals", "🎯 Prediction"])

# --- Tab 1: Data Overview ---
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# --- Tab 2: Visualizations ---
with tab2:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
    st.pyplot(plt)

    st.subheader("Feature Distributions")
    selected_col = st.selectbox("Select Column to Visualize", df.columns)
    plt.figure(figsize=(8, 4))
    sns.histplot(df[selected_col], kde=True, color='green')
    st.pyplot(plt)

# --- Tab 3: Prediction ---
with tab3:
    st.subheader("🧪 Predict Sustainability")

    input_df = user_input_features()
    st.write("### 🔍 Input Values", input_df)

    try:
        prediction = model.predict(input_df)
        st.success(f"🌿 Predicted Sustainability Score/Label: **{prediction[0]}**")
    except Exception as e:
        st.error("❌ Prediction failed. Please check input format and feature names.")

# Footer
st.markdown("---")
st.caption("Made with 💚 by Pralhad Balaji Jadhav | Model: Logistic Regression")
