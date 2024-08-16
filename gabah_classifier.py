import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Label mapping
label_mapping = {
    0: 'ir64'
    1: 'pandan wangi'
    2: 'rojo lele'
}

# Function to perform prediction based on selected model and input data
def predict(model, input_data):
    prediction = model.predict(input_data)
    predicted_label = [label_mapping.get(pred) for pred in prediction]
    probabilities = model.predict_proba(input_data)
    return predicted_label, probabilities

# Streamlit interface
st.title('Gabah Classifier')

model_choice = st.sidebar.selectbox('Select Model', ('Decision Tree (Recommendation)',
                                                     'XGBoost',
                                                     'Random Forest'))

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    is_file_uploaded = True
else:
    def user_input_features():
        gender_encoded = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        refund = st.number_input('Refund', min_value=0, max_value=9999999, value=0)
        wallet_balance = st.number_input('Wallet Balance', min_value=0, max_value=9999999, value=0)
        # List of products
        products = [
            'None', 'Man Fashion', 'Woman Fashion', 'Food & Drink', 'Ride Hailing',
            'Keperluan Rumah Tangga', 'Travel', 'Keperluan Anak', 'Elektronik', 'Other',
            'Transportasi (Kereta Pesawat Kapal)', 'Top Up Game', 'Otomotif', 'Pulsa',
            'Kesehatan', 'Investasi', 'Sewa Motor/Mobil', 'Hotel', 'Tagihan (WIFI PLN)'
        ]

        # Function to format product names
        def product_format_func(x):
            return products[x]

        # Slider for most bought product
        most_bought_product = st.selectbox(
            'Most Bought Product',
            options=list(range(len(products))),
            format_func=product_format_func
        )

        st.write(f'You selected: {products[most_bought_product]}')

        total_gross_amount = st.number_input('Total Gross Amount', min_value=0, max_value=99999999, value=0)
        total_discount_amount = st.number_input('Total Discount Amount', min_value=0, max_value=99999999, value=0)
        recency = st.number_input('Recency', min_value=0, max_value=1000, value=0)
        frequency = st.number_input('Frequency', min_value=0, max_value=1000, value=0)
        monetary = st.number_input('Monetary', min_value=0, max_value=999999999, value=0)

        data = {'gender_encoded': gender_encoded,
                'refund': refund,
                'wallet_balance': wallet_balance,
                'most_bought_product': most_bought_product,
                'total_gross_amount': total_gross_amount,
                'total_discount_amount': total_discount_amount,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
                }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    is_file_uploaded = False

# Combine user input with the cleaned df

## Load the cleaned data
df_raw = pd.read_csv('cleaned_df.csv')
df_independent = df_raw.drop(columns=['customer_tier_encoded'])

# Clean column names in df_independent from extra spaces
df_independent.columns = df_independent.columns.str.strip()

# Combine user input with the cleaned df
df = pd.concat([input_df, df_independent], axis=0)

# Selects only the user input data
if is_file_uploaded:
    df_input = input_df
else:
    df_input = df.iloc[:1, :]

# Display the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df_input)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df_input)

## Load the scaler
min_max_scaler = joblib.load('min_max_scaler.joblib')
std_scaler = joblib.load('std_scaler.joblib')

df_input.loc[:, ["refund", "wallet_balance", "total_gross_amount",
         "total_discount_amount", "monetary"]] = min_max_scaler.transform(df_input[["refund", 
         "wallet_balance", "total_gross_amount",
         "total_discount_amount", "monetary"]])

df_std = std_scaler.transform(df_input)

# Load the models
loaded_xgb = joblib.load('xgb_model.joblib')
loaded_dt = joblib.load('decision_tree_model.joblib')
loaded_rf = joblib.load('randomforest_model.joblib')

# Perform prediction on button click
if st.sidebar.button('Predict'):
    if model_choice == 'XGBoost':
        result, probabilities = predict(loaded_xgb, df_std)
    elif model_choice == 'Decision Tree (Recommendation)':
        result, probabilities = predict(loaded_dt, df_std)
    elif model_choice == 'Random Forest':
        result, probabilities = predict(loaded_rf, df_std)

    st.sidebar.write(f'Predictions:')
    for i, (res, prob) in enumerate(zip(result, probabilities)):
        st.sidebar.write(f'Row {i+1}: {res}')
        st.sidebar.markdown('**Prediction Probabilities:**')
        probabilities_html = "<ul>"
        for j, p in enumerate(prob):
            probabilities_html += f"<li style='font-size: 12px;'>{label_mapping[j]}: {p:.4f}</li>"
        probabilities_html += "</ul>"
        st.sidebar.markdown(probabilities_html, unsafe_allow_html=True)