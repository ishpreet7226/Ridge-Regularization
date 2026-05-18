# importing libraries
import streamlit as st
import matplotlib.pyplot as plt

# importing regression function
from model import regression

# streamlit title
st.title("Ridge Regularization on Fish Market Dataset")

# sidebar sliders
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

epochs = st.sidebar.slider("Epochs", 100, 5000, 1000)

penalty = st.sidebar.slider("Penalty", 0.1, 10.0, 1.0)

# calling regression function
w, b, mse, y_test, y_pred = regression(
    lr,
    epochs,
    penalty
)

# displaying weights
st.subheader("Weights")
st.write(w)

# displaying bias
st.subheader("Bias")
st.write(b)

# displaying mse
st.subheader("Mean Squared Error")
st.write(mse)

# graph plotting
fig, ax = plt.subplots()

# actual values
ax.plot(y_test, label="Actual Values")

# predicted values
ax.plot(y_pred, label="Predicted Values")

# graph legend
ax.legend()

# showing graph
st.pyplot(fig)
