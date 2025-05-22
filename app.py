import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_model():
    with open("banglore_house_model.pkl", "rb") as file:
        model = pkl.load(file)
    return model

def preprocess_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df.drop(columns=["society"], inplace=True)
    df.dropna(inplace=True)
    df["total_sqft"] = df["total_sqft"].str.replace(r'\s+', '', regex=True)
    df["total_sqft"] = df["total_sqft"].str.split('-').str[-1]
    df["total_sqft"] = df["total_sqft"].str.replace(r'[^0-9.]', '', regex=True)
    df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")
    df.dropna(inplace=True)
    df["size"] = df["size"].str.split(" ").str[0].astype(float)
    df["price_per_bhk"] = df["price"] / df["size"]
    
    ms = MinMaxScaler()
    df[["total_sqft", "size"]] = ms.fit_transform(df[["total_sqft", "size"]])
    
    df = pd.get_dummies(df, columns=["area_type", "location", "availability"], drop_first=True)
    feature_cols = df.drop(columns=["price"]).columns.tolist()
    return df, feature_cols, ms

def main():
    st.title("🏠 Bengaluru House Price Prediction")
    st.markdown("Enter the details below to get an estimated house price prediction.")
    
    df, feature_cols, ms = preprocess_data()
    locations = [col.replace("location_", "") for col in df.columns if "location_" in col]
    area_types = [col.replace("area_type_", "") for col in df.columns if "area_type_" in col]
    
    total_sqft = st.slider("Total Sqft", min_value=500, max_value=10000, value=1000, step=50)
    size = st.selectbox("Size (BHK)", range(1, 11), index=1)
    price_per_bhk = st.slider("Price per BHK (₹)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    location = st.selectbox("Location", locations)
    area_type = st.selectbox("Area Type", area_types)
    
    user_input_df = pd.DataFrame([[total_sqft, size, price_per_bhk]], columns=["total_sqft", "size", "price_per_bhk"])
    user_input_df[["total_sqft", "size"]] = ms.transform(user_input_df[["total_sqft", "size"]])
    
    user_input = {col: 0 for col in feature_cols}
    user_input["total_sqft"], user_input["size"], user_input["price_per_bhk"] = user_input_df.iloc[0]
    
    if f"location_{location}" in user_input:
        user_input[f"location_{location}"] = 1
    if f"area_type_{area_type}" in user_input:
        user_input[f"area_type_{area_type}"] = 1
    
    input_data = pd.DataFrame([user_input])
    input_data = input_data.reindex(columns=feature_cols, fill_value=0)
    
    model = load_model()
    if st.button("💰 Predict Price"):
        prediction = model.predict(input_data)
        st.success(f"🏡 Estimated House Price: ₹{prediction[0]:,.2f}")
        
if __name__ == "__main__":
    main()
