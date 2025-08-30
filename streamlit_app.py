import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Global Sample Data for Demonstration ---
SAMPLE_CSV_CONTENT = """angle_of_attack,reynolds_number,Cl,Cd,shape_name
0,300000,0.45,0.007,NACA 4412
2,300000,0.65,0.008,NACA 4412
4,300000,0.85,0.012,NACA 4412
6,300000,1.0,0.018,NACA 4412
0,300000,0.0,0.006,NACA 0012
2,300000,0.23,0.0065,NACA 0012
4,300000,0.45,0.008,NACA 0012
6,300000,0.68,0.01,NACA 0012
0,300000,0.2,0.005,Eppler 423
2,300000,0.5,0.006,Eppler 423
4,300000,0.8,0.008,Eppler 423
6,300000,1.1,0.012,Eppler 423
"""

# --- Model Training Function ---
def train_and_evaluate_model(uploaded_file):
    """
    Trains a Random Forest Regressor model from an uploaded CSV file.
    Args:
        uploaded_file: The file object from st.file_uploader.
    Returns:
        A tuple containing the trained model, a status message,
        the loaded dataframe, the list of feature names,
        and the list of unique shapes.
    """
    if uploaded_file is None:
        st.error("Please upload a CSV file to train the model.")
        return None, "Please upload a CSV file to train the model.", None, None, []
    
    try:
        # Use st.spinner for a visual loading indicator
        with st.spinner("Loading data and training model..."):
            data = pd.read_csv(uploaded_file)
            
            required_columns = ['angle_of_attack', 'reynolds_number', 'Cl', 'Cd', 'shape_name']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                st.error(f"Error: Missing required columns in CSV: {', '.join(missing)}")
                return None, f"Error: Missing required columns in CSV: {', '.join(missing)}", None, None, []
            
            data_encoded = pd.get_dummies(data, columns=['shape_name'], prefix='shape')
            
            feature_names = ['angle_of_attack', 'reynolds_number'] + [col for col in data_encoded.columns if 'shape_' in col]
            X = data_encoded[feature_names]
            y = data_encoded[['Cl', 'Cd']]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse_cl = np.sqrt(mean_squared_error(y_test['Cl'], y_pred[:, 0]))
            rmse_cd = np.sqrt(mean_squared_error(y_test['Cd'], y_pred[:, 1]))
            
            unique_shapes = data['shape_name'].unique().tolist()
            
            status_message = f"Model trained successfully. 🎉\n\n**RMSE for $C_l$:** {rmse_cl:.4f}\n**RMSE for $C_d$:** {rmse_cd:.4f}"
            st.success("Model trained successfully!")
            
            # Return the model, data, feature names, and unique shapes
            return model, status_message, data, feature_names, unique_shapes
    
    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        return None, f"An error occurred during training: {e}", None, None, []

# --- Streamlit UI ---

st.set_page_config(page_title="AI Surrogate CFD Tool", layout="wide")

st.title("AI Surrogate CFD Tool")
st.markdown("### Upload and Train Your Model")

# Streamlit uses session state to store variables that persist across reruns.
# We'll use this to store the trained model, data, and feature names.
if "model" not in st.session_state:
    st.session_state.model = None
if "data" not in st.session_state:
    st.session_state.data = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "shapes" not in st.session_state:
    st.session_state.shapes = []

# Column for file uploader and buttons
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload a CSV file with CFD data", type="csv")
    
    # Check if a file is already in the session state to pre-fill the uploader
    if uploaded_file is None and "sample_data_loaded" in st.session_state and st.session_state.sample_data_loaded:
        uploaded_file = st.session_state.uploaded_file

# Load sample data logic
with col2:
    if st.button("Load Sample Data"):
        csv_file = io.StringIO(SAMPLE_CSV_CONTENT)
        st.session_state.uploaded_file = csv_file
        st.session_state.sample_data_loaded = True
        st.rerun() # Rerun the app to process the loaded data

# The training button in Streamlit
if st.button("Train Model"):
    model, status, data, feature_names, shapes = train_and_evaluate_model(uploaded_file)
    st.session_state.model = model
    st.session_state.data = data
    st.session_state.feature_names = feature_names
    st.session_state.shapes = shapes
    st.session_state.status_message = status

# Display training status if available
if "status_message" in st.session_state and st.session_state.status_message:
    st.markdown("### Training Status")
    st.markdown(st.session_state.status_message)

st.divider()

st.markdown("### CFD Predictor")

# Main prediction interface
col3, col4 = st.columns(2)
with col3:
    # Use st.session_state to control the interactivity of the widgets
    interactive = st.session_state.model is not None
    
    angle_of_attack = st.slider("Angle of Attack ($^\circ$)", -10.0, 10.0, 0.0, step=0.1, disabled=not interactive)
    reynolds_number = st.slider("Reynolds Number", 100000.0, 1000000.0, 300000.0, step=100000.0, disabled=not interactive)
    
    # Use a selectbox for shapes. The options come from the trained data.
    if st.session_state.shapes:
        shape_name = st.selectbox("Airfoil Name", options=st.session_state.shapes, disabled=not interactive)
    else:
        st.selectbox("Airfoil Name", options=[], disabled=True, help="Train a model first to populate this list.")
    
    predict_button = st.button("Predict", disabled=not interactive)

with col4:
    # Display prediction results and plot
    st.markdown("#### Prediction Results")
    cl_placeholder = st.empty()
    cd_placeholder = st.empty()
    plot_placeholder = st.empty()

# Prediction logic is now an event handler for the button click
if predict_button and st.session_state.model:
    model = st.session_state.model
    data = st.session_state.data
    feature_names = st.session_state.feature_names
    
    # Create input DataFrame and align columns
    input_df = pd.DataFrame([[angle_of_attack, reynolds_number, shape_name]],
                            columns=['angle_of_attack', 'reynolds_number', 'shape_name'])
    
    input_encoded = pd.get_dummies(input_df, columns=['shape_name'], prefix='shape')
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_encoded)
    cl_pred = prediction[0][0]
    cd_pred = prediction[0][1]
    
    # Display predictions
    cl_placeholder.info(f"Predicted Lift Coefficient ($C_l$): **{cl_pred:.4f}**")
    cd_placeholder.info(f"Predicted Drag Coefficient ($C_d$): **{cd_pred:.4f}**")
    
    # Generate and display plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    shape_data = data[data['shape_name'] == shape_name]
    
    # Plot Lift Coefficient
    axs[0].scatter(shape_data['angle_of_attack'], shape_data['Cl'], alpha=0.5, label='Training Data')
    axs[0].scatter([angle_of_attack], [cl_pred], color='red', s=100, label='Prediction')
    axs[0].set_title(f'Lift Coefficient ($C_l$) Prediction for {shape_name}')
    axs[0].set_xlabel('Angle of Attack ($^\circ$)')
    axs[0].set_ylabel('Lift Coefficient')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Drag Coefficient
    axs[1].scatter(shape_data['angle_of_attack'], shape_data['Cd'], alpha=0.5, label='Training Data')
    axs[1].scatter([angle_of_attack], [cd_pred], color='red', s=100, label='Prediction')
    axs[1].set_title(f'Drag Coefficient ($C_d$) Prediction for {shape_name}')
    axs[1].set_xlabel('Angle of Attack ($^\circ$)')
    axs[1].set_ylabel('Drag Coefficient')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plot_placeholder.pyplot(fig)
