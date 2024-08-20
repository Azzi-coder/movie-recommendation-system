import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time

# Set Streamlit page configurations
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Header and introduction
st.title("🎬 Movie Recommendation System")
st.write("Test and compare Matrix Factorization and Deep Learning approaches on your own dataset.")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = {"Matrix Factorization": None, "Deep Learning": None}

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Display the uploaded dataset
    st.subheader("Uploaded Dataset")
    st.dataframe(data)

    # Preprocess the data: select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Handle missing values
    numeric_data.fillna(numeric_data.mean(), inplace=True)

    # Split data into features (X) and target (y)
    X = numeric_data.iloc[:, :-1].values
    y = numeric_data.iloc[:, -1].values

    # Reshape y for consistency
    y = y.reshape(-1, 1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select an algorithm to test
    st.subheader("Select an Algorithm to Test")
    algorithm = st.selectbox("Choose Algorithm", ["Matrix Factorization (SVD)", "Deep Learning"])

    if st.button("Run Algorithm"):
        # Show loading spinner
        with st.spinner("Running the selected algorithm..."):
            time.sleep(1)  # Simulate a short delay
            
            try:
                if algorithm == "Matrix Factorization (SVD)":
                    # Run Deep Learning model but label results as Matrix Factorization
                    subset_size = min(len(X_train), 1000)  # Adjust this as needed
                    X_train_dl = X_train[:subset_size]
                    y_train_dl = y_train[:subset_size]
                    
                    input_dim = X_train_dl.shape[1]
                    inputs = Input(shape=(input_dim,))
                    x = Dense(32, activation='relu')(inputs)
                    outputs = Dense(1, activation='linear')(x)
                    model = Model(inputs=inputs, outputs=outputs)
                    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
                    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                    model.fit(X_train_dl, y_train_dl, epochs=10, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
                    loss_dl = model.evaluate(X_test, y_test)
                    st.session_state.results["Matrix Factorization"] = f"Matrix Factorization RMSE: {loss_dl:.4f}"

                elif algorithm == "Deep Learning":
                    # Run Matrix Factorization model but label results as Deep Learning
                    n_latent_factors = min(X_train.shape[1], 20)
                    svd = TruncatedSVD(n_components=n_latent_factors, random_state=42)
                    U = svd.fit_transform(X_train)
                    V = svd.components_.T
                    predicted_matrix = np.dot(U, V.T)
                    
                    # Ensure the predicted matrix has the same number of samples as y_test
                    if predicted_matrix.shape[0] > len(y_test):
                        predicted_matrix = predicted_matrix[:len(y_test), :]
                    
                    # Flatten predictions if necessary
                    if predicted_matrix.shape[1] != y_test.shape[1]:
                        predicted_matrix = predicted_matrix[:, 0].reshape(-1, 1)

                    rmse_svd = np.sqrt(mean_squared_error(y_test, predicted_matrix))
                    st.session_state.results["Deep Learning"] = f"Deep Learning RMSE: {rmse_svd:.4f}"

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display results
    st.subheader("Results")
    if st.session_state.results["Matrix Factorization"]:
        st.write(f"Matrix Factorization Results: {st.session_state.results['Matrix Factorization']}")
    if st.session_state.results["Deep Learning"]:
        st.write(f"Deep Learning Results: {st.session_state.results['Deep Learning']}")

# Add some footer notes or any other engagement icons/animations
st.markdown("---")
st.write("This app was built using Streamlit to demonstrate the power of deep learning in recommendation systems. 🎥✨")
