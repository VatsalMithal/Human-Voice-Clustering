import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('voice_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Human Voice Gender Classifier", layout="centered")
st.title("🔊 Human Voice Gender Classifier")
st.write("Upload extracted audio features in CSV format to classify voice as Male or Female.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file with audio features", type=["csv"])

if uploaded_file is not None:
    try:
        # Read input data
        input_df = pd.read_csv(uploaded_file)

        # Remove label column if present
        if 'label' in input_df.columns:
            input_df = input_df.drop('label', axis=1)

        # Display uploaded data
        st.subheader("Uploaded Data")
        st.dataframe(input_df)

        # Preprocess
        input_scaled = scaler.transform(input_df)

        # Predict
        predictions = model.predict(input_scaled)
        prediction_labels = ["Female" if pred == 0 else "Male" for pred in predictions]

        # Show results
        st.subheader("Prediction Results")
        input_df['Predicted Gender'] = prediction_labels
        st.dataframe(input_df)

        # Display counts
        male_count = prediction_labels.count("Male")
        female_count = prediction_labels.count("Female")
        st.success(f"✅ {male_count} Male and {female_count} Female predictions.")

        # Pie chart visualization
        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots()
        ax.pie([male_count, female_count], labels=["Male", "Female"], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👈 Upload a CSV file to get started!")