import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
import os

# Load the saved models and encoders
@st.cache_resource
def load_models():
    models = {
        'Random Forest': joblib.load('saved_models/random_forest.joblib'),
        'SVM': joblib.load('saved_models/svm.joblib'),
        'KNN': joblib.load('saved_models/knn.joblib'),
        'Decision Tree': joblib.load('saved_models/decision_tree.joblib'),
        'Gradient Boosting': joblib.load('saved_models/gradient_boosting.joblib'),
        'AdaBoost': joblib.load('saved_models/adaboost.joblib'),
        'Extra Trees': joblib.load('saved_models/extra_trees.joblib'),
        'Gaussian Naive Bayes': joblib.load('saved_models/gaussian_naive_bayes.joblib'),
        'Logistic Regression': joblib.load('saved_models/logistic_regression.joblib'),
        'Neural Network': joblib.load('saved_models/neural_network.joblib')
    }
    nn_model = load_model('saved_models/neural_network.h5')
    scaler = joblib.load('saved_models/scaler.joblib')
    label_encoder = joblib.load('saved_models/label_encoder.joblib')
    return models, nn_model, scaler, label_encoder

models, nn_model, scaler, label_encoder = load_models()

def extract_color_features(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    color_features = cv2.normalize(hist, hist).flatten()
    return color_features

def extract_texture_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    texture_features = graycoprops(glcm, 'contrast').flatten()
    return texture_features

def extract_shape_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        aspect_ratio = float(img.shape[1]) / img.shape[0]
        shape_features = [area, perimeter, aspect_ratio]
    else:
        shape_features = [0, 0, 0]
    
    return shape_features

def extract_visual_features(img):
    color_features = extract_color_features(img)
    texture_features = extract_texture_features(img)
    shape_features = extract_shape_features(img)
    features = np.concatenate((color_features, texture_features, shape_features))
    return features

def predict_rice_type(img, model_name):
    features = extract_visual_features(img)
    features_scaled = scaler.transform([features])
    
    if model_name == 'Neural Network (Custom)':
        prediction = nn_model.predict(features_scaled)
        predicted_class = np.argmax(prediction)
        probabilities = prediction[0]
    else:
        model = models[model_name]
        predicted_class = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
    
    rice_type = label_encoder.inverse_transform([predicted_class])[0]
    return rice_type, probabilities

st.title('Rice Type Classifier')

uploaded_file = st.file_uploader("Choose an image of rice", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    model_name = st.selectbox(
        'Select the model for prediction',
        ('Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Gradient Boosting', 
         'AdaBoost', 'Extra Trees', 'Gaussian Naive Bayes', 'Logistic Regression', 
         'Neural Network', 'Neural Network (Custom)')
    )
    
    if st.button('Predict'):
        rice_type, probabilities = predict_rice_type(img, model_name)
        st.success(f'The predicted rice type is: {rice_type}')
        
        st.write('Prediction Probabilities:')
        probabilities_df = pd.DataFrame({
            'Rice Type': label_encoder.classes_,
            'Probability': probabilities
        })
        probabilities_df = probabilities_df.sort_values('Probability', ascending=False)
        st.dataframe(probabilities_df.style.format({'Probability': '{:.4f}'}))
