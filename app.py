import streamlit as st
import numpy as np
from model import train

#Title
st.title("Linear Regression AI")
st.subheader("Simple Regression Model")

#Trained model
model=train()

#sidebar
st.sidebar.header("Input features")
input_values=st.sidebar.slider("Select the value of x",1,20,1)
#Prediction
input_array=np.array([[input_values]])
Prediction=model.predict(input_array)

#Displaying the result we get
st.write(f'### Input value : {input_values}')
st.write(f'### Output value : {Prediction[0]:.2f}')