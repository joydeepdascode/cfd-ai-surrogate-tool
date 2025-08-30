import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import altair as alt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.errors import EmptyDataError
import joblib

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="AI Surrogate CFD Tool", layout="wide")

# --- Global Sample Data for Demonstration ---
SAMPLE_CSV_CONTENT = """angle_of_attack,reynolds_number,Cl,Cd,shape_name
0,300000,0.45,0.007,NACA 4412
2,300000,0.65,0.008,NACA 4412
4,300000,0.85,0.012,NACA 4412
6,300000,1.0,0.018,NACA 4412
8,300000,1.1,0.024,NACA 4412
10,300000,1.2,0.035,NACA 4412
12,300000,1.15,0.05,NACA 4412
14,300000,1.0,0.07,NACA 4412
-2,300000,0.2,0.008,NACA 4412
-4,300000,-0.1,0.012,NACA 4412
0,300000,0.0,0.006,NACA 0012
2,300000,0.23,0.0065,NACA 0012
4,300000,0.45,0.008,NACA 0012
6,300000,0.68,0.01,NACA 0012
8,300000,0.8,0.014,NACA 0012
10,300000,0.95,0.02,NACA 0012
-2,300000,-0.2,0.0065,NACA 0012
-4,300000,-0.4,0.008,NACA 0012
0,300000,0.2,0.005,Eppler 423
2,300000,0.5,0.006,Eppler 423
4,300000,0.8,0.008,Eppler 423
6,300000,1.1,0.012,Eppler 423
8,300000,1.35,0.02,Eppler 423
10,300000,1.5,0.035,Eppler 423
-2,300000,-0.1,0.007,Eppler 423
-4,300000,-0.4,0.01,Eppler 423
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

# Add a session state for user-uploaded airfoils
if "user_airfoils" not in st.session_state:
    st.session_state.user_airfoils = {}

def parse_airfoil_file(uploaded_file):
    """
    Parses an uploaded file object (.dat or .txt) into a NumPy array.
    
    Args:
        uploaded_file: A Streamlit UploadedFile object.
    
    Returns:
        A NumPy array of shape (N, 2) containing the x, y coordinates,
        or None if parsing fails.
    """
    try:
        # Decode the file content as a string
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.strip().split('\n')
        
        # Heuristically skip header lines
        data_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that aren't likely coordinate data
            if not line or not line[0].isdigit() and not line.startswith('-'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Attempt to convert to float
                    x = float(parts[0])
                    y = float(parts[1])
                    data_lines.append([x, y])
                except ValueError:
                    # Skip lines that can't be parsed as floats
                    continue

        if not data_lines:
            st.error("Error: Could not find valid coordinate data in the uploaded file.")
            return None
            
        return np.array(data_lines)
    except Exception as e:
        st.error(f"An error occurred while parsing the airfoil file: {e}")
        return None

def get_airfoil_coords(shape_name):
    """
    Fetches airfoil coordinates from either hard-coded or user-uploaded data.
    """
    if shape_name in AIRFOIL_COORDS:
        coords = AIRFOIL_COORDS[shape_name]
        lines = coords.strip().split('\n')
        data = [list(map(float, line.split())) for line in lines]
        return np.array(data)
    elif shape_name in st.session_state.user_airfoils:
        return st.session_state.user_airfoils[shape_name]
    return None

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
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
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

# --- Model Training Function with Cross-Validation ---
def train_and_evaluate_model(uploaded_file, model_choice, hyperparameters, scaler_choice):
    """
    Trains a selected machine learning model from an uploaded CSV file using
    K-Fold Cross-Validation for robust evaluation.
    """
    if uploaded_file is None:
        st.error("Please upload a CSV file to train the model.")
        return None, "Please upload a CSV file to train the model.", None, None, [], None
    
    try:
        with st.spinner("Loading data and training model..."):
            data = pd.read_csv(uploaded_file)
            
            required_columns = ['angle_of_attack', 'reynolds_number', 'Cl', 'Cd', 'shape_name']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                st.error(f"Error: Missing required columns in CSV: {', '.join(missing)}")
                return None, f"Error: Missing required columns in CSV: {', '.join(missing)}", None, None, [], None
            
            data_encoded = pd.get_dummies(data, columns=['shape_name'], prefix='shape')
            
            feature_names = ['angle_of_attack', 'reynolds_number'] + [col for col in data_encoded.columns if 'shape_' in col]
            X = data_encoded[feature_names]
            y = data_encoded[['Cl', 'Cd']]
            
            # --- Scaling Logic ---
            scaler = None
            if scaler_choice == 'MinMax':
                scaler = MinMaxScaler()
            elif scaler_choice == 'Standard':
                scaler = StandardScaler()
            
            if scaler:
                # Fit the scaler to the numerical features (excluding one-hot encoded ones)
                numerical_features = ['angle_of_attack', 'reynolds_number']
                X[numerical_features] = scaler.fit_transform(X[numerical_features])
            
            # Use K-Fold Cross-Validation
            n_folds = 5
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            models = {}
            r2_scores_cl = []
            r2_scores_cd = []
            rmse_scores_cl = []
            rmse_scores_cd = []
            
            fold_results = []

            if model_choice == 'Random Forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **hyperparameters)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    r2_scores_cl.append(r2_score(y_test['Cl'], y_pred[:, 0]))
                    r2_scores_cd.append(r2_score(y_test['Cd'], y_pred[:, 1]))
                    rmse_scores_cl.append(np.sqrt(mean_squared_error(y_test['Cl'], y_pred[:, 0])))
                    rmse_scores_cd.append(np.sqrt(mean_squared_error(y_test['Cd'], y_pred[:, 1])))

                # Retrain the model on the full dataset for prediction
                model.fit(X, y)
                final_model = model
            else:
                if model_choice == 'Gradient Boosting':
                    model_cl = GradientBoostingRegressor(random_state=42, **hyperparameters)
                    model_cd = GradientBoostingRegressor(random_state=42, **hyperparameters)
                elif model_choice == 'Support Vector Machine':
                    model_cl = SVR(**hyperparameters)
                    model_cd = SVR(**hyperparameters)
                elif model_choice == 'Neural Network (MLP)':
                    model_cl = MLPRegressor(random_state=42, **hyperparameters)
                    model_cd = MLPRegressor(random_state=42, **hyperparameters)
                
                # Cross-validate the Cl model
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train_cl, y_test_cl = y['Cl'].iloc[train_index], y['Cl'].iloc[test_index]
                    
                    model_cl.fit(X_train, y_train_cl)
                    y_pred_cl = model_cl.predict(X_test)
                    r2_scores_cl.append(r2_score(y_test_cl, y_pred_cl))
                    rmse_scores_cl.append(np.sqrt(mean_squared_error(y_test_cl, y_pred_cl)))
                
                # Cross-validate the Cd model
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train_cd, y_test_cd = y['Cd'].iloc[train_index], y['Cd'].iloc[test_index]
                    
                    model_cd.fit(X_train, y_train_cd)
                    y_pred_cd = model_cd.predict(X_test)
                    r2_scores_cd.append(r2_score(y_test_cd, y_pred_cd))
                    rmse_scores_cd.append(np.sqrt(mean_squared_error(y_test_cd, y_pred_cd)))
                    
                # Retrain both models on the full dataset for prediction
                model_cl.fit(X, y['Cl'])
                model_cd.fit(X, y['Cd'])
                final_model = {'Cl': model_cl, 'Cd': model_cd}

            avg_r2_cl = np.mean(r2_scores_cl)
            std_r2_cl = np.std(r2_scores_cl)
            avg_r2_cd = np.mean(r2_scores_cd)
            std_r2_cd = np.std(r2_scores_cd)
            
            avg_rmse_cl = np.mean(rmse_scores_cl)
            std_rmse_cl = np.std(rmse_scores_cl)
            avg_rmse_cd = np.mean(rmse_scores_cd)
            std_rmse_cd = np.std(rmse_scores_cd)
            
            unique_shapes = data['shape_name'].unique().tolist()
            
            status_message = f"""
            Model trained successfully! 🎉
            **Model:** {model_choice}
            **Scaling:** {scaler_choice}
            **Cross-Validation Results ({n_folds}-Fold):**
            **Average R-squared ($R^2$) Score:**
            $C_l$: {avg_r2_cl:.4f} (± {std_r2_cl:.4f})
            $C_d$: {avg_r2_cd:.4f} (± {std_r2_cd:.4f})
            
            **Average Root Mean Squared Error (RMSE):**
            $C_l$: {avg_rmse_cl:.4f} (± {std_rmse_cl:.4f})
            $C_d$: {avg_rmse_cd:.4f} (± {std_rmse_cd:.4f})
            """
            st.success("Model training complete!")
            
            return final_model, status_message, data, feature_names, unique_shapes, scaler
    
    except EmptyDataError:
        st.error("Error: The uploaded CSV file is empty or has no data to parse. Please check the file.")
        return None, "Error: The uploaded CSV file is empty or has no data to parse.", None, None, [], None
    except Exception as e:
        st.error(f"An unexpected error occurred during training: {e}")
        return None, f"An unexpected error occurred during training: {e}", None, None, [], None

# --- Main Streamlit UI ---
st.title("AI Surrogate CFD Tool")
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
if "scaler" not in st.session_state: st.session_state.scaler = None
if "user_airfoils" not in st.session_state: st.session_state.user_airfoils = {}

# ----------------------------------------------------------------
## 1. Upload Data & Train Model

st.header("1. Upload Data & Train Model", divider="blue")

col_upload, col_sample, col_load_model = st.columns([3, 1, 2])

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

with col_load_model:
    st.markdown("##### Or load a saved model:")
    loaded_model_file = st.file_uploader(
        "Load a saved model (.joblib)",
        type="joblib",
        help="Upload a previously saved model file to bypass training."
    )
    if loaded_model_file:
        try:
            with st.spinner("Loading model..."):
                loaded_state = joblib.load(loaded_model_file)
                # Restore session state from the loaded dictionary
                for key, value in loaded_state.items():
                    st.session_state[key] = value
                st.success("Model and settings loaded successfully! ✅")
        except Exception as e:
            st.error(f"Error loading model: {e}. Please ensure the file is a valid .joblib file.")
            st.session_state.model = None

# --- Model Selection and Hyperparameters ---
st.subheader("Model Selection & Hyperparameters")
model_choice = st.selectbox(
    "Select a Regression Model",
    options=['Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Neural Network (MLP)'],
    help="Choose the machine learning algorithm to train as the surrogate model."
)
scaler_choice = st.selectbox(
    "Select a Data Scaler",
    options=['None', 'MinMax', 'Standard'],
    help="Scaling can improve the performance of some models like SVR and MLP."
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
        model, status, data, feature_names, shapes, scaler = train_and_evaluate_model(file_to_train, model_choice, hyperparameters, scaler_choice)
        st.session_state.model = model
        st.session_state.data = data
        st.session_state.feature_names = feature_names
        st.session_state.shapes = shapes
        st.session_state.status_message = status
        st.session_state.scaler = scaler
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

# Add "Save Model" button
if st.session_state.model is not None:
    # Package all relevant session state variables into a dictionary to save
    model_state_to_save = {
        "model": st.session_state.model,
        "feature_names": st.session_state.feature_names,
        "shapes": st.session_state.shapes,
        "status_message": st.session_state.status_message,
        "data": st.session_state.data,
        "scaler": st.session_state.scaler
    }
    
    # Serialize the model to a byte stream
    saved_model_bytes = io.BytesIO()
    joblib.dump(model_state_to_save, saved_model_bytes)
    
    st.download_button(
        label="💾 Save Model",
        data=saved_model_bytes.getvalue(),
        file_name="ai_surrogate_cfd_model.joblib",
        mime="application/octet-stream",
        help="Download the trained model and its associated data.",
        use_container_width=True
    )

st.divider()

# ----------------------------------------------------------------
## 2. CFD Predictor

st.header("2. CFD Predictor", divider="green")

# --- New Airfoil Upload Section ---
st.subheader("Add Custom Airfoil Geometry")
airfoil_upload_col, airfoil_name_col = st.columns([2, 1])

with airfoil_upload_col:
    uploaded_airfoil_file = st.file_uploader(
        "Upload a custom airfoil geometry (.dat or .txt)",
        type=["dat", "txt"],
        help="Upload a file with x, y coordinates separated by a space or tab. Header lines will be skipped."
    )

with airfoil_name_col:
    custom_airfoil_name = st.text_input(
        "Name for the new airfoil",
        placeholder="e.g., My-Custom-Airfoil"
    )

if uploaded_airfoil_file and custom_airfoil_name:
    if custom_airfoil_name in AIRFOIL_COORDS or custom_airfoil_name in st.session_state.user_airfoils:
        st.warning(f"An airfoil named '{custom_airfoil_name}' already exists. Please choose a different name.")
    else:
        parsed_coords = parse_airfoil_file(uploaded_airfoil_file)
        if parsed_coords is not None:
            st.session_state.user_airfoils[custom_airfoil_name] = parsed_coords
            st.success(f"Successfully added custom airfoil: **{custom_airfoil_name}**! 🎉")
            # Clear the uploader and text input to prevent re-addition on refresh
            uploaded_airfoil_file = None
            st.rerun()

tab_predict, tab_info = st.tabs(["Make Prediction", "Prediction Info"])

with tab_predict:
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Input Parameters")
        interactive = st.session_state.model is not None
        
        if not interactive:
            st.warning("☝️ Train or load a model in Section 1 to enable prediction inputs.")

        angle_of_attack = st.slider(
            "Angle of Attack ($^\circ$)",
            -10.0, 15.0, 0.0, step=0.1,
            disabled=not interactive,
            help="The angle at which the airfoil meets the oncoming air."
        )
        reynolds_number = st.slider(
            "Reynolds Number",
            100000.0, 1000000.0, 300000.0, step=100000.0,
            disabled=not interactive,
            help="A dimensionless quantity used to predict flow patterns."
        )
        
        # Combine hard-coded and user-uploaded airfoils
        airfoil_options = list(AIRFOIL_COORDS.keys()) + list(st.session_state.user_airfoils.keys())
        
        if st.session_state.shapes or airfoil_options:
            shape_name = st.selectbox(
                "Airfoil Name",
                options=st.session_state.shapes if st.session_state.shapes else airfoil_options,
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
            scaler = st.session_state.scaler
            
            input_df = pd.DataFrame([[angle_of_attack, reynolds_number, shape_name]],
                                     columns=['angle_of_attack', 'reynolds_number', 'shape_name'])
            
            input_encoded = pd.get_dummies(input_df, columns=['shape_name'], prefix='shape')
            input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
            
            # --- Apply scaler to the input data for prediction ---
            if scaler:
                numerical_features = ['angle_of_attack', 'reynolds_number']
                # Use .values to avoid potential DataFrame indexing issues with a single row
                input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features].values)

            
            # Use the correct prediction logic based on the model type
            if isinstance(model, dict):
                cl_pred = model['Cl'].predict(input_encoded)[0]
                cd_pred = model['Cd'].predict(input_encoded)[0]
            else:
                prediction = model.predict(input_encoded)
                cl_pred = prediction[0][0]
                cd_pred = prediction[0][1]
            
            col_cl, col_cd = cl_output_container.columns(2)
            with col_cl:
                st.metric(label="Lift Coefficient ($C_l$)", value=f"{cl_pred:.4f}")
            with col_cd:
                st.metric(label="Drag Coefficient ($C_d$)", value=f"{cd_pred:.4f}")
            
            # --- START OF MODIFIED PLOTTING CODE ---
            if data is not None and shape_name in data['shape_name'].unique():
                shape_data = data[data['shape_name'] == shape_name].copy()
                shape_data['Type'] = 'Training Data'
                
                # Create a DataFrame for the single prediction point
                prediction_df = pd.DataFrame({
                    'angle_of_attack': [angle_of_attack],
                    'Cl': [cl_pred],
                    'Cd': [cd_pred],
                    'Type': ['Prediction']
                })
                
                # Layer the training and prediction points for the Cl plot
                cl_chart_training = alt.Chart(shape_data).mark_circle(size=60).encode(
                    x=alt.X('angle_of_attack', title='Angle of Attack ($^\circ$)'),
                    y=alt.Y('Cl', title='Lift Coefficient ($C_l$)'),
                    tooltip=['angle_of_attack', 'Cl', 'Cd']
                ).properties(
                    title=f'Lift Coefficient ($C_l$) Prediction for {shape_name}'
                )
                
                cl_chart_prediction = alt.Chart(prediction_df).mark_point(color='red', size=200, shape='diamond').encode(
                    x='angle_of_attack',
                    y='Cl',
                    tooltip=['angle_of_attack', 'Cl', 'Cd']
                )
                
                cl_chart = cl_chart_training + cl_chart_prediction
                
                # Layer the training and prediction points for the Cd plot
                cd_chart_training = alt.Chart(shape_data).mark_circle(size=60).encode(
                    x=alt.X('angle_of_attack', title='Angle of Attack ($^\circ$)'),
                    y=alt.Y('Cd', title='Drag Coefficient ($C_d$)'),
                    tooltip=['angle_of_attack', 'Cl', 'Cd']
                ).properties(
                    title=f'Drag Coefficient ($C_d$) Prediction for {shape_name}'
                )

                cd_chart_prediction = alt.Chart(prediction_df).mark_point(color='red', size=200, shape='diamond').encode(
                    x='angle_of_attack',
                    y='Cd',
                    tooltip=['angle_of_attack', 'Cl', 'Cd']
                )

                cd_chart = cd_chart_training + cd_chart_prediction
                
                # Display the charts side-by-side using columns
                col_cl_chart, col_cd_chart = st.columns(2)
                with col_cl_chart:
                    st.altair_chart(cl_chart, use_container_width=True)
                with col_cd_chart:
                    st.altair_chart(cd_chart, use_container_width=True)
            else:
                st.warning("No training data available for this airfoil shape to plot against. The prediction is still valid.")
            # --- END OF MODIFIED PLOTTING CODE ---
            
with tab_info:
    st.markdown("""
        #### How Predictions Work
        After training the machine learning model on your provided CFD data,
        this tool uses the trained model to predict the Lift and Drag Coefficients
        ($C_l$ and $C_d$) for new input parameters (Angle of Attack, Reynolds Number, and Airfoil Shape).
        
        For models like **Random Forest**, a single model is trained to predict both $C_l$ and $C_d$ simultaneously.
        
        For models like **Gradient Boosting**, **SVR**, and **MLP**, two separate models are trained: one for $C_l$ and one for $C_d$. The predictions are then combined.
        
        The model uses **one-hot encoding** internally to handle different airfoil shapes.
        
        #### Understanding the Plots
        The plots visualize the predicted coefficients against the training data for the selected airfoil shape.
        * **Red diamond**: Represents the current prediction.
        * **Blue circles**: Show the historical training data points for the chosen airfoil.
        
        This helps in understanding how the prediction fits within the known data range.
    """)
    
st.divider()

# ----------------------------------------------------------------
## 3. Visualize Model Behavior
st.header("3. Visualize Model Behavior", divider="rainbow")

if st.session_state.model is not None and len(st.session_state.shapes) > 0:
    st.markdown("""
    This section allows you to visualize the surrogate model's predictions over a **continuous range**
    of a single parameter while keeping others constant.
    """)
    
    # New UI Section: Sliders and Dropdown for range prediction
    col_vis_1, col_vis_2 = st.columns(2)

    with col_vis_1:
        vis_shape_name = st.selectbox(
            "Select Airfoil for Visualization",
            options=st.session_state.shapes
        )
        
        vis_reynolds_number = st.slider(
            "Constant Reynolds Number",
            min_value=100000.0,
            max_value=1000000.0,
            value=300000.0,
            step=100000.0
        )
        
    with col_vis_2:
        vis_angle_range = st.slider(
            "Angle of Attack Range ($^\circ$)",
            min_value=-10.0,
            max_value=15.0,
            value=(-5.0, 15.0),
            step=0.1
        )
        vis_num_points = st.slider(
            "Number of Prediction Points",
            min_value=10,
            max_value=200,
            value=100,
            step=10
        )

    vis_button = st.button("📈 Generate Prediction Curves", use_container_width=True)
    
    if vis_button:
        # Generate DataFrame with a sequence of values
        angle_start, angle_end = vis_angle_range
        angle_points = np.linspace(angle_start, angle_end, vis_num_points)
        
        # Keep other parameters constant
        pred_data = {
            'angle_of_attack': angle_points,
            'reynolds_number': [vis_reynolds_number] * vis_num_points,
            'shape_name': [vis_shape_name] * vis_num_points
        }
        
        pred_df = pd.DataFrame(pred_data)
        
        # One-hot encode the DataFrame
        pred_encoded = pd.get_dummies(pred_df, columns=['shape_name'], prefix='shape')
        
        # Ensure the columns match the trained model's feature names
        pred_encoded = pred_encoded.reindex(columns=st.session_state.feature_names, fill_value=0)
        
        # Apply the scaler if one was used during training
        if st.session_state.scaler:
            numerical_features = ['angle_of_attack', 'reynolds_number']
            pred_encoded[numerical_features] = st.session_state.scaler.transform(pred_encoded[numerical_features])
        
        # Batch prediction
        with st.spinner("Generating predictions..."):
            model = st.session_state.model
            if isinstance(model, dict):
                cl_preds = model['Cl'].predict(pred_encoded)
                cd_preds = model['Cd'].predict(pred_encoded)
                pred_df['Cl_pred'] = cl_preds
                pred_df['Cd_pred'] = cd_preds
            else:
                batch_prediction = model.predict(pred_encoded)
                pred_df['Cl_pred'] = batch_prediction[:, 0]
                pred_df['Cd_pred'] = batch_prediction[:, 1]
                
            # Filter the original training data for the selected shape
            training_data = st.session_state.data[st.session_state.data['shape_name'] == vis_shape_name].copy()
            training_data['Type'] = 'Training Data'
            
            # Create a combined DataFrame for plotting
            pred_df['Type'] = 'Predicted Curve'

            # Plotting with Altair
            # CL Plot
            cl_line_chart = alt.Chart(pred_df).mark_line(color='green').encode(
                x=alt.X('angle_of_attack', title='Angle of Attack ($^\circ$)'),
                y=alt.Y('Cl_pred', title='Lift Coefficient ($C_l$)')
            ).properties(
                title=f'$C_l$ Prediction Curve for {vis_shape_name} at Re = {vis_reynolds_number:.0f}'
            )

            cl_scatter_chart = alt.Chart(training_data).mark_circle(color='blue', size=60).encode(
                x=alt.X('angle_of_attack'),
                y=alt.Y('Cl'),
                tooltip=['angle_of_attack', 'Cl', 'Cd']
            )
            
            final_cl_chart = (cl_line_chart + cl_scatter_chart).interactive()

            # CD Plot
            cd_line_chart = alt.Chart(pred_df).mark_line(color='orange').encode(
                x=alt.X('angle_of_attack', title='Angle of Attack ($^\circ$)'),
                y=alt.Y('Cd_pred', title='Drag Coefficient ($C_d$)')
            ).properties(
                title=f'$C_d$ Prediction Curve for {vis_shape_name} at Re = {vis_reynolds_number:.0f}'
            )

            cd_scatter_chart = alt.Chart(training_data).mark_circle(color='red', size=60).encode(
                x=alt.X('angle_of_attack'),
                y=alt.Y('Cd'),
                tooltip=['angle_of_attack', 'Cl', 'Cd']
            )

            final_cd_chart = (cd_line_chart + cd_scatter_chart).interactive()
            
            # Display charts side-by-side
            st.success("Plots generated!")
            col_chart_cl, col_chart_cd = st.columns(2)
            with col_chart_cl:
                st.altair_chart(final_cl_chart, use_container_width=True)
            with col_chart_cd:
                st.altair_chart(final_cd_chart, use_container_width=True)
