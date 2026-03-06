import pickle
import pandas as pd
import streamlit as st

# Load model
with open("Linear_model.pkl","rb") as file:
    model = pickle.load(file)

# Load encoder
with open("label_encoder.pkl","rb") as file1:
    encoder = pickle.load(file1)

# Load data
df = pd.read_csv("cleaned_data.csv")



st.set_page_config(
    page_title="House Price Prediction Bangalore",
    page_icon=r"C:\Users\Hp\Downloads\house_logo.png"
)
with st.sidebar:
    st.title("Bengalore House Price Prediction")
    st.image("https://thumbs.dreamstime.com/b/green-houses-community-model-abstract-real-estate-logo-vector-professional-architecture-company-design-115154734.jpg")

#input fields

#'location','bhk','total_sqft','bath','encoded_loc'
location = st.selectbox("Location: ",options=df["location"].unique())

bhk = st.selectbox("bhk: ",options=sorted(df["bhk"].unique()))

total_sqft = st.number_input("Total_sqft: ",min_value=300)

bath = st.selectbox("No. of Restrooms: ",options=sorted(df["bath"].unique()))

#encoded the new location
encoded_loc = encoder.transform([location])

#new data preparation
new_data = [[bhk,total_sqft,bath,encoded_loc[0]]]

#prediction
col1,col2 = st.columns([1,2])

if col2.button("Predict House Price"):
    pred = model.predict(new_data)[0]
    pred = round(pred*100000)
    st.subheader(f"Predicted Price : Rs. {pred}")
