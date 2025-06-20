
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load models and scalers
model_full = joblib.load('voice_classifier_model.pkl')
scaler_full = joblib.load('scaler.pkl')
model_10 = joblib.load('model_10features.pkl')
scaler_10 = joblib.load('scaler_10.pkl')

df = pd.read_csv('vocal_gender_features_new.csv')

st.set_page_config(page_title="Human Voice App", layout="wide")

with st.sidebar:
    selected = st.selectbox("📍 Select a Page", [
        "🏠 Introduction", "📊 EDA", "🔮 Prediction (10 Features)",
        "📁 CSV Prediction (43 Features)", "📌 Clustering", "👤 About"
    ])

# Page: Introduction
if selected == "🏠 Introduction":
    st.title("🎤 Human Voice Gender Classification and Clustering")
    st.markdown("""
    This project classifies and clusters human voices using audio features.
    - **Data Source:** Extracted MFCC, spectral, pitch features from audio.
    - **Models:** Random Forest, SVM, MLP for classification.
    - **Clustering:** KMeans, DBSCAN with Silhouette score and PCA.
    - **UI:** Built using Streamlit.
    """)
    st.image("https://img.freepik.com/free-vector/voice-assistant-concept-illustration_114360-3413.jpg", use_container_width=True)

# Page: EDA
elif selected == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("## Gender Distribution")
    st.bar_chart(df['label'].value_counts())

    st.write("## Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    st.write("## Selected Feature Distributions")
    selected_features = ['mean_pitch', 'rms_energy', 'mfcc_1_mean']
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=feature, kde=True, hue='label', ax=ax)
        st.pyplot(fig)

# Page: Prediction with 10 features
elif selected == "🔮 Prediction (10 Features)":
    st.title("🔮 Gender Prediction from 10 Features")

    with st.form("predict_form"):
        st.write("### Enter Audio Feature Values:")
        features_10 = [
            ('mean_pitch', 300, 2500),
            ('rms_energy', 0.02, 0.2),
            ('mfcc_1_mean', -50, 50),
            ('mfcc_2_mean', -50, 50),
            ('mfcc_3_mean', -50, 50),
            ('mean_spectral_centroid', 700, 3000),
            ('mean_spectral_bandwidth', 1100, 2000),
            ('spectral_kurtosis', -2, 10),
            ('log_energy', -5, 5),
            ('zero_crossing_rate', 0.01, 0.3)
        ]
        user_input = []
        for feature, min_val, max_val in features_10:
            val = st.slider(f"{feature} ({min_val}-{max_val})", float(min_val), float(max_val), float((min_val+max_val)/2))
            user_input.append(val)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_scaled = scaler_10.transform([user_input])
        pred = model_10.predict(input_scaled)[0]
        gender = "Female" if pred == 0 else "Male"
        st.success(f"🎯 Predicted Gender: {gender}")

# Page: Full 43-feature CSV Prediction
elif selected == "📁 CSV Prediction (43 Features)":
    st.title("📁 Gender Prediction from Full Feature CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if 'label' in input_df.columns:
                input_df = input_df.drop(columns=['label'])
            st.write("Uploaded Data:")
            st.dataframe(input_df.head())

            input_scaled = scaler_full.transform(input_df)
            predictions = model_full.predict(input_scaled)
            input_df['Predicted Gender'] = ["Female" if i == 0 else "Male" for i in predictions]

            st.write("Prediction Results:")
            st.dataframe(input_df)

            male = list(predictions).count(1)
            female = list(predictions).count(0)
            st.success(f"✅ {male} Male and {female} Female predictions")

            fig, ax = plt.subplots()
            ax.pie([male, female], labels=["Male", "Female"], autopct="%1.1f%%", colors=["skyblue", "pink"])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Page: Clustering
elif selected == "📌 Clustering":
    st.title("📌 Clustering Analysis")

    X = df.drop(columns=["label"])
    X_scaled = scaler_full.transform(X)

    st.write("### KMeans Silhouette Scores")
    scores = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(range(2, 6), scores, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

    st.write("### PCA Visualization (K=2)")
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", ax=ax)
    st.pyplot(fig)

# Page: About
elif selected == "👤 About":
    st.title("👤 About This Project")
    st.markdown("""
    **Created By:** Vatsal Mithal  
    **Description:** ML + Streamlit project with both manual and CSV-based prediction modes for human voice  
    **Technologies Used:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn
    """)
