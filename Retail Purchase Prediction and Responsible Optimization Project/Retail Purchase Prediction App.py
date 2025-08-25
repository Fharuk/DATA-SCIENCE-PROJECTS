# -----------------------------
# Streamlit App for Retail Purchase Prediction
# -----------------------------
import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# Load saved models
ebm_pipeline = joblib.load("ebm_pipeline.pkl")
cat_model = CatBoostRegressor()
cat_model.load_model("catboost_model.cbm")

st.title("Retail Purchase Prediction App")
st.write("Predict the purchase amount for a customer using EBM or CatBoost.")

# -----------------------------
# User Input
# -----------------------------
def user_input_features():
    User_ID = st.number_input("User ID", min_value=1000000, max_value=1100000, value=1000001)
    Product_ID = st.text_input("Product ID", "P00069042")
    Gender = st.selectbox("Gender", ["F", "M"])
    Age = st.selectbox("Age", [0, 1, 6, 11, 16])  # numeric age after preprocessing
    Occupation = st.selectbox("Occupation", list(range(21)))
    City_Category = st.selectbox("City Category", ["A", "B", "C"])
    Stay_In_Current_City_Years = st.number_input("Years in current city", min_value=0, max_value=10, value=2)
    Marital_Status = st.selectbox("Marital Status", [0, 1])
    Product_Category_1 = st.selectbox("Product Category 1", list(range(1, 21)))
    Product_Category_2 = st.selectbox("Product Category 2", list(range(-1, 19)))
    Product_Category_3 = st.selectbox("Product Category 3", list(range(-1, 19)))
    Product_Category_2_missing = 1 if Product_Category_2 == -1 else 0
    Product_Category_3_missing = 1 if Product_Category_3 == -1 else 0
    user_txn_count = st.number_input("User transaction count", min_value=0, value=1)
    user_avg_purchase = st.number_input("User avg purchase", min_value=0, value=1000)
    user_total_purchase = st.number_input("User total purchase", min_value=0, value=1000)
    product_buyer_count = st.number_input("Product buyer count", min_value=0, value=1)
    product_avg_purchase = st.number_input("Product avg purchase", min_value=0, value=1000)
    Product_ID_freq = st.number_input("Product ID frequency", min_value=0, value=1)
    User_ID_freq = st.number_input("User ID frequency", min_value=0, value=1)
    City_ProductCat1 = f"{City_Category}_{Product_Category_1}"
    Occupation_ProductCat1 = f"{Occupation}_{Product_Category_1}"

    # Construct DataFrame
    data = {
        'Gender': Gender,
        'Age': Age,
        'Occupation': Occupation,
        'City_Category': City_Category,
        'Stay_In_Current_City_Years': Stay_In_Current_City_Years,
        'Marital_Status': Marital_Status,
        'Product_Category_1': Product_Category_1,
        'Product_Category_2': Product_Category_2,
        'Product_Category_3': Product_Category_3,
        'Product_Category_2_missing': Product_Category_2_missing,
        'Product_Category_3_missing': Product_Category_3_missing,
        'user_txn_count': user_txn_count,
        'user_avg_purchase': user_avg_purchase,
        'user_total_purchase': user_total_purchase,
        'product_buyer_count': product_buyer_count,
        'product_avg_purchase': product_avg_purchase,
        'Product_ID_freq': Product_ID_freq,
        'User_ID_freq': User_ID_freq,
        'City_ProductCat1': City_ProductCat1,
        'Occupation_ProductCat1': Occupation_ProductCat1
    }

    df = pd.DataFrame(data, index=[0])

    # Convert all categorical columns to string to match CatBoost training
    cat_columns = [
        'Gender', 'Occupation', 'City_Category', 'Marital_Status',
        'Product_Category_1', 'Product_Category_2', 'Product_Category_3',
        'City_ProductCat1', 'Occupation_ProductCat1'
    ]
    for col in cat_columns:
        df[col] = df[col].astype(str)

    return df

input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
model_choice = st.radio("Select Model", ["EBM", "CatBoost"])

if st.button("Predict"):
    if model_choice == "EBM":
        prediction = ebm_pipeline.predict(input_df)
    else:
        # For CatBoost, input DataFrame must have same column types as training
        prediction = cat_model.predict(input_df)

    st.success(f"Predicted Purchase Amount: ₦{prediction[0]:,.2f}")
