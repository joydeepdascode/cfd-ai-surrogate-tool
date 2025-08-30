import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pandas.errors import EmptyDataError

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="AI Surrogate CFD Tool", layout="wide")

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

# --- Hard-coded Airfoil Coordinates for Demonstration ---
AIRFOIL_COORDS = {
    "NACA 4412": """1.0000 0.0000
0.9500 0.0097
0.9000 0.0195
0.8500 0.0292
0.8000 0.0388
0.7500 0.0483
0.7000 0.0575
0.6000 0.0754
0.5000 0.0911
0.4000 0.1044
0.3000 0.1147
0.2000 0.1213
0.1000 0.1232
0.0000 0.1189
-0.1000 0.1069
-0.2000 0.0886
-0.3000 0.0657
-0.4000 0.0392
-0.5000 0.0102
-0.6000 -0.0198
-0.7000 -0.0514
-0.8000 -0.0838
-0.9000 -0.1167
-1.0000 -0.1500""",
    "NACA 0012": """1.0000 0.0000
0.9500 0.0060
0.9000 0.0085
0.8500 0.0104
0.8000 0.0120
0.7500 0.0134
0.7000 0.0145
0.6000 0.0163
0.5000 0.0175
0.4000 0.0182
0.3000 0.0185
0.2000 0.0182
0.1000 0.0172
0.0000 0.0152
-0.1000 0.0123
-0.2000 0.0085
-0.3000 0.0040
-0.4000 -0.0008
-0.5000 -0.0062
-0.6000 -0.0123
-0.7000 -0.0190
-0.8000 -0.0264
-0.9000 -0.0345
-1.0000 -0.0434""",
    "Eppler 423": """1.0000 0.0000
0.9500 0.0120
0.9000 0.0200
0.8500 0.0260
0.8000 0.0310
0.7500 0.0350
0.7000 0.0380
0.6000 0.0420
0.5000 0.0450
0.4000 0.0460
0.3000 0.0450
0.2000 0.0420
0.1000 0.0360
0.0000 0.0280
-0.1000 0.0180
-0.2000 0.0060
-0.3000 -0.0080
-0.4000 -0.0240
-0.5000 -0.0410
-0.6000 -0.0600
-0.7000 -0.0810
-0.8000 -0.1030
-0.9000 -0.1260
-1.0000 -0.1500"""
}

def get_airfoil_coords(shape_name):
    """Parses hard-coded airfoil coordinate string into a NumPy array."""
    if shape_name not in AIRFOIL_COORDS:
        return None
    coords = AIRFOIL_COORDS[shape_name]
    lines = coords.strip().split('\n')
    data = [list(map(float, line.split())) for line in lines]
    return np.array(data)

def plot_airfoil(coords, angle_of_attack):
    """Plots a single airfoil, rotated by a given angle of attack."""
    if coords is None:
        return None
    
    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(angle_of_attack)
    
    # Create a 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Apply rotation to the coordinates
    rotated_coords = np.dot(coords, rotation_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rotated_coords[:, 0], rotated_coords[:, 1], color='black')
    
    ax.set_title(f'Airfoil at {angle_of_attack}° Angle of Attack')
    ax.set_xlabel('X/c')
    ax.set_ylabel('Y/c')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Draw a line representing the freestream flow direction
    ax.arrow(0.0, 0.0, 1.0, 0.0, head_width=0.03, head_length=0.05, fc='blue', ec='blue', lw=1, zorder=3)
    ax.text(0.5, -0.1, "Freestream Flow", color='blue', ha='center', va='top')
    
    plt.tight_layout()
    return fig

# --- Model Training Function ---
def train_and_evaluate_model(uploaded_file, model_choice, hyperparameters):
    """
    Trains a selected machine learning model from an uploaded CSV file.
    """
    if uploaded_file is None:
        st.error("Please upload a CSV file to train the model.")
        return None, "Please upload a CSV file to train the model.", None, None, []
    
    try:
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
            
            # --- Dynamically select and create the model ---
            if model_choice == 'Random Forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **hyperparameters)
            elif model_choice == 'Gradient Boosting':
                model = GradientBoostingRegressor(random_state=42, **hyperparameters)
            elif model_choice == 'Support Vector Machine':
                model = SVR(**hyperparameters)
            elif model_choice == 'Neural Network (MLP)':
                # MLP needs specific handling for multi-output regression
                # This is a simplified example, a more robust solution is required for production
                model = MLPRegressor(random_state=42, **hyperparameters)
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse_cl = np.sqrt(mean_squared_error(y_test['Cl'], y_pred[:, 0]))
            rmse_cd = np.sqrt(mean_squared_error(y_test['Cd'], y_pred[:, 1]))
            r2_cl = r2_score(y_test['Cl'], y_pred[:, 0])
            r2_cd = r2_score(y_test['Cd'], y_pred[:, 1])
            
            unique_shapes = data['shape_name'].unique().tolist()
            
            status_message = f"""
            Model trained successfully! 🎉
            **Model:** {model_choice}
            **Root Mean Squared Error (RMSE):**
            $C_l$: {rmse_cl:.4f}
            $C_d$: {rmse_cd:.4f}
            **R-squared ($R^2$) Score:**
            $C_l$: {r2_cl:.4f}
            $C_d$: {r2_cd:.4f}
            """
            st.success("Model training complete!")
            
            return model, status_message, data, feature_names, unique_shapes
    
    except EmptyDataError:
        st.error("Error: The uploaded CSV file is empty or has no data to parse. Please check the file.")
        return None, "Error: The uploaded CSV file is empty or has no data to parse.", None, None, []
    except Exception as e:
        st.error(f"An unexpected error occurred during training: {e}")
        return None, f"An unexpected error occurred during training: {e}", None, None, []

# --- Main Streamlit UI ---
st.title("🚀 AI Surrogate CFD Tool")
st.markdown("""
    This application utilizes **Machine Learning** to create a surrogate model for Computational Fluid Dynamics (CFD) data. 
    Upload your airfoil data, train a model, and then make quick predictions for Lift ($C_l$) and Drag ($C_d$) Coefficients.
""")

st.info("💡 **Getting Started:** Upload your CSV data or load our sample data to train the model. Then, use the sliders and dropdown to make predictions!")

# Initialize session state variables if they don't exist
if "model" not in st.session_state: st.session_state.model = None
if "data" not in st.session_state: st.session_state.data = None
if "feature_names" not in st.session_state: st.session_state.feature_names = None
if "shapes" not in st.session_state: st.session_state.shapes = []
if "status_message" not in st.session_state: st.session_state.status_message = ""
if "sample_data_loaded" not in st.session_state: st.session_state.sample_data_loaded = False
if "uploaded_file_object" not in st.session_state: st.session_state.uploaded_file_object = None

# ----------------------------------------------------------------
## 1. Upload Data & Train Model

st.header("1. Upload Data & Train Model", divider="blue")

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file_content = st.file_uploader(
        "Upload a CSV file with CFD data", 
        type="csv",
        help="Ensure your CSV has columns: 'angle_of_attack', 'reynolds_number', 'Cl', 'Cd', 'shape_name'."
    )
    if uploaded_file_content:
        st.session_state.uploaded_file_object = uploaded_file_content
        st.session_state.sample_data_loaded = False

with col_sample:
    st.markdown("##### Or try with sample data:")
    if st.button("Load Sample Data 📊", use_container_width=True):
        csv_file_io = io.StringIO(SAMPLE_CSV_CONTENT)
        st.session_state.uploaded_file_object = csv_file_io
        st.session_state.sample_data_loaded = True
        st.rerun()

# --- Model Selection and Hyperparameters ---
st.subheader("Model Selection & Hyperparameters")
model_choice = st.selectbox(
    "Select a Regression Model",
    options=['Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Neural Network (MLP)'],
    help="Choose the machine learning algorithm to train as the surrogate model."
)

hyperparameters = {}
with st.expander("Adjust Model Hyperparameters"):
    if model_choice == 'Random Forest':
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, 10)
        max_depth = st.slider('Max Depth', 1, 30, None, 1)
        hyperparameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
    elif model_choice == 'Gradient Boosting':
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, 10)
        learning_rate = st.slider('Learning Rate', 0.01, 0.5, 0.1, 0.01)
        max_depth = st.slider('Max Depth', 1, 10, 3, 1)
        hyperparameters = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
    elif model_choice == 'Support Vector Machine':
        C = st.slider('C (Regularization)', 0.1, 100.0, 1.0, 0.1)
        epsilon = st.slider('Epsilon', 0.01, 1.0, 0.1, 0.01)
        kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly'])
        hyperparameters = {'C': C, 'epsilon': epsilon, 'kernel': kernel}
    elif model_choice == 'Neural Network (MLP)':
        hidden_layer_sizes = st.slider('Hidden Layers', 1, 10, 2, 1)
        # Simplified to a single layer for demonstration
        hyperparameters = {'hidden_layer_sizes': (hidden_layer_sizes,), 'max_iter': 500}
        st.warning("Neural networks may require more data and careful tuning.")


# The training button
if st.button("✨ Train Model", type="primary", use_container_width=True):
    file_to_train = None
    if st.session_state.uploaded_file_object:
        file_to_train = st.session_state.uploaded_file_object
    elif st.session_state.sample_data_loaded:
        file_to_train = io.StringIO(SAMPLE_CSV_CONTENT)

    if file_to_train:
        model, status, data, feature_names, shapes = train_and_evaluate_model(file_to_train, model_choice, hyperparameters)
        st.session_state.model = model
        st.session_state.data = data
        st.session_state.feature_names = feature_names
        st.session_state.shapes = shapes
        st.session_state.status_message = status
    else:
        st.warning("Please upload a CSV file or load sample data before training.")

# Display training status
if st.session_state.status_message:
    st.subheader("Training Results")
    if "Error" in st.session_state.status_message:
        st.error(st.session_state.status_message)
    else:
        st.success("Model Training Complete!")
        with st.expander("View Training Details"):
            st.markdown(st.session_state.status_message)

st.divider()

# ----------------------------------------------------------------
## 2. CFD Predictor

st.header("2. CFD Predictor", divider="green")

tab_predict, tab_info = st.tabs(["Make Prediction", "Prediction Info"])

with tab_predict:
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Input Parameters")
        interactive = st.session_state.model is not None
        
        if not interactive:
            st.warning("☝️ Train a model in Section 1 to enable prediction inputs.")

        angle_of_attack = st.slider(
            "Angle of Attack ($^\circ$)", 
            -10.0, 10.0, 0.0, step=0.1, 
            disabled=not interactive,
            help="The angle at which the airfoil meets the oncoming air."
        )
        reynolds_number = st.slider(
            "Reynolds Number", 
            100000.0, 1000000.0, 300000.0, step=100000.0, 
            disabled=not interactive,
            help="A dimensionless quantity used to predict flow patterns."
        )
        
        if st.session_state.shapes:
            shape_name = st.selectbox(
                "Airfoil Name", 
                options=st.session_state.shapes, 
                disabled=not interactive,
                help="Select the airfoil shape for prediction."
            )
        else:
            shape_name = st.selectbox("Airfoil Name", options=["No airfoils available"], disabled=True, help="Train a model first to populate this list.")
        
        predict_button = st.button("🔍 Predict Coefficients", type="secondary", use_container_width=True, disabled=not interactive)

    with col_output:
        st.subheader("Airfoil Visualization")
        
        # Get airfoil coordinates and plot dynamically
        coords = get_airfoil_coords(shape_name)
        if coords is not None:
            airfoil_fig = plot_airfoil(coords, angle_of_attack)
            st.pyplot(airfoil_fig)
            plt.close(airfoil_fig)
        else:
            st.warning("Airfoil geometry data not available for this shape.")
            
        st.subheader("Predicted Results")
        
        cl_output_container = st.empty()
        cd_output_container = st.empty()
        plot_output_container = st.empty()

        if predict_button and st.session_state.model:
            model = st.session_state.model
            data = st.session_state.data
            feature_names = st.session_state.feature_names
            
            input_df = pd.DataFrame([[angle_of_attack, reynolds_number, shape_name]],
                                     columns=['angle_of_attack', 'reynolds_number', 'shape_name'])
            
            input_encoded = pd.get_dummies(input_df, columns=['shape_name'], prefix='shape')
            input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
            
            prediction = model.predict(input_encoded)
            # Handle multi-output for SVR and MLPRegressor
            if model_choice in ['Support Vector Machine', 'Neural Network (MLP)']:
                cl_pred, cd_pred = prediction[0]
            else:
                cl_pred, cd_pred = prediction[0][0], prediction[0][1]
            
            col_cl, col_cd = cl_output_container.columns(2)
            with col_cl:
                st.metric(label="Lift Coefficient ($C_l$)", value=f"{cl_pred:.4f}")
            with col_cd:
                st.metric(label="Drag Coefficient ($C_d$)", value=f"{cd_pred:.4f}")
            
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            shape_data = data[data['shape_name'] == shape_name]
            
            axs[0].scatter(shape_data['angle_of_attack'], shape_data['Cl'], alpha=0.5, label='Training Data')
            axs[0].scatter([angle_of_attack], [cl_pred], color='red', s=100, label='Prediction')
            axs[0].set_title(f'Lift Coefficient ($C_l$) Prediction for {shape_name}')
            axs[0].set_xlabel('Angle of Attack ($^\circ$)')
            axs[0].set_ylabel('Lift Coefficient')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].scatter(shape_data['angle_of_attack'], shape_data['Cd'], alpha=0.5, label='Training Data')
            axs[1].scatter([angle_of_attack], [cd_pred], color='red', s=100, label='Prediction')
            axs[1].set_title(f'Drag Coefficient ($C_d$) Prediction for {shape_name}')
            axs[1].set_xlabel('Angle of Attack ($^\circ$)')
            axs[1].set_ylabel('Drag Coefficient')
            axs[1].legend()
            axs[1].grid(True)
            
            plt.tight_layout()
            plot_output_container.pyplot(fig)
            plt.close(fig)

with tab_info:
    st.markdown("""
        #### How Predictions Work
        After training the Random Forest Regressor model on your provided CFD data, 
        this tool uses the trained model to predict the Lift and Drag Coefficients 
        ($C_l$ and $C_d$) for new input parameters (Angle of Attack, Reynolds Number, and Airfoil Shape).
        
        The model uses **one-hot encoding** internally to handle different airfoil shapes.
        
        #### Understanding the Plots
        The plots visualize the predicted coefficients against the training data for the selected airfoil shape.
        * **Red dot**: Represents the current prediction.
        * **Blue dots**: Show the historical training data points for the chosen airfoil.
        
        This helps in understanding how the prediction fits within the known data range.
    """)

st.divider()
st.caption(f"App developed for AI Surrogate CFD Tool. Current location: Kolkata, West Bengal, India. © 2025")
