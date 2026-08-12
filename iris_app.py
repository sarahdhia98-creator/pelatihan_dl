import streamlit as st
import pickle
import numpy as np
from tensorflow import keras

# ==========================
# Load deployment bundle
# ==========================
with open("iris_deployment_bundle.pkl", "rb") as file:
    bundle = pickle.load(file)

# Reconstruct Keras model
model = keras.models.model_from_json(bundle["model_json"])
model.set_weights(bundle["model_weights"])

# Load preprocessing objects
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]

# ==========================
# Prediction Function
# ==========================
def make_prediction(features):

    # Convert to numpy array
    input_array = np.array(features).reshape(1, -1)

    # Scale input
    X_scaled = scaler.transform(input_array)

    # Predict probabilities
    probability = model.predict(X_scaled, verbose=0)

    # Get predicted class index
    class_index = np.argmax(probability, axis=1)

    # Convert to species name
    prediction = label_encoder.inverse_transform(class_index)

    return prediction[0]

# ==========================
# Streamlit Interface
# ==========================
def main():

    st.title("Iris Flower Prediction")

    st.write("Enter the flower measurements below.")

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

    if st.button("Make Prediction"):

        features = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]

        prediction = make_prediction(features)

        st.success(f" Predicted Species: **{prediction}**")


if __name__ == "__main__":
    main()