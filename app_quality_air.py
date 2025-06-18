# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    model_data = joblib.load('air_quality_model.pkl')
    return model_data

model_data = load_model()
model = model_data['model']
le = model_data['label_encoder']
feature_names = model_data['feature_names']
original_data = model_data['original_data']
resampled_data = model_data['resampled_data']

# Reverse mapping for categories
category_mapping = {
    0: 'BAIK',
    1: 'SEDANG',
    2: 'TIDAK SEHAT'
}

# App title
st.title('Air Quality Index Prediction')
st.write("""
This app predicts the air quality category based on pollutant measurements using a Gradient Boosting model.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    pm10 = st.sidebar.slider('PM10', float(original_data['pm10'].min()), float(original_data['pm10'].max()), float(original_data['pm10'].median()))
    so2 = st.sidebar.slider('SO2', float(original_data['so2'].min()), float(original_data['so2'].max()), float(original_data['so2'].median()))
    co = st.sidebar.slider('CO', float(original_data['co'].min()), float(original_data['co'].max()), float(original_data['co'].median()))
    o3 = st.sidebar.slider('O3', float(original_data['o3'].min()), float(original_data['o3'].max()), float(original_data['o3'].median()))
    no2 = st.sidebar.slider('NO2', float(original_data['no2'].min()), float(original_data['no2'].max()), float(original_data['no2'].median()))
    
    data = {
        'pm10': pm10,
        'so2': so2,
        'co': co,
        'o3': o3,
        'no2': no2
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write(f"Predicted Air Quality Category: **{category_mapping[prediction[0]]}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame({
    'Category': ['BAIK', 'SEDANG', 'TIDAK SEHAT'],
    'Probability': prediction_proba[0]
})
st.write(proba_df)

# Model evaluation
st.subheader('Model Evaluation')
st.write("""
The Gradient Boosting model was evaluated with 10-fold cross validation:
- Average Accuracy: ~0.95
- Average Precision (weighted): ~0.95
- Average Recall (weighted): ~0.95
- Average F1-score (weighted): ~0.95
""")

# Data distribution
st.subheader('Data Distribution')

col1, col2 = st.columns(2)

with col1:
    st.write("**Original Data Distribution**")
    fig, ax = plt.subplots()
    original_data['categori'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(['BAIK', 'SEDANG', 'TIDAK SEHAT'], rotation=0)
    st.pyplot(fig)

with col2:
    st.write("**Resampled Data Distribution**")
    fig, ax = plt.subplots()
    resampled_data['categori'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(['BAIK', 'SEDANG', 'TIDAK SEHAT'], rotation=0)
    st.pyplot(fig)

# Feature importance
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
st.pyplot(fig)

st.write(feature_importance)