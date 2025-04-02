import streamlit as st
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load("knn.pkl")

# Streamlit UI
st.title("Iris Flower Prediction using KNN")
st.write("Enter the flower attributes to predict the species")

# Input fields for user
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"The predicted species is: {species_map[prediction[0]]}")
