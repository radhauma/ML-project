import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/rf_iris.pkl")

st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower measurements:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width,
                            petal_length, petal_width]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Flower: {prediction[0]}")