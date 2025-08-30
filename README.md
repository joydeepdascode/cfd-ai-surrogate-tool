# AI Surrogate CFD Tool 🚀

This is a Streamlit application that serves as an AI-powered surrogate model for Computational Fluid Dynamics (CFD) simulations. By training a **Random Forest Regressor** on historical CFD data, the app can quickly predict the Lift ($C\_l$) and Drag ($C\_d$) coefficients for new input parameters, eliminating the need for time-consuming and computationally expensive full-scale simulations.

The app provides a simple, intuitive interface for data scientists, engineers, and students to build and test their own surrogate models.

-----

## Features

  * **Interactive UI:** A user-friendly interface built with Streamlit for easy interaction.
  * **Model Training:** Train a `RandomForestRegressor` model by uploading a CFD data CSV file.
  * **Dynamic Visualization:** Visualize the selected airfoil and its orientation as you adjust the Angle of Attack.
  * **Quick Predictions:** Get instant predictions for Lift ($C\_l$) and Drag ($C\_d$) coefficients based on the trained model.
  * **Performance Metrics:** View the model's performance with a Root Mean Squared Error (RMSE) score after training.
  * **Sample Data:** A "Load Sample Data" button to get started immediately without needing to find a dataset.

-----

## How it Works

The application uses a machine learning approach to **approximate** the results of a complex CFD simulation.

1.  **Data Ingestion:** The app reads a CSV file containing CFD simulation results. The data must include `angle_of_attack`, `reynolds_number`, `Cl`, `Cd`, and `shape_name`.
2.  **Preprocessing:** The airfoil shapes are converted into a numerical format using **one-hot encoding**, allowing the model to understand different geometries.
3.  **Model Training:** A `RandomForestRegressor` model is trained on the preprocessed data to learn the relationship between the input parameters (angle of attack, Reynolds number, and shape) and the output coefficients ($C\_l$ and $C\_d$).
4.  **Prediction:** Once trained, the model can take new input parameters and predict the corresponding coefficients almost instantaneously, providing a huge speed-up over traditional CFD.

-----

## Getting Started

### Prerequisites

  * Python 3.7+
  * `pip` package manager

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file should contain:*
    ```
    streamlit
    numpy
    pandas
    scikit-learn
    matplotlib
    ```

### Running the App

1.  Make sure you are in the project directory.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The app will automatically open in your web browser.

-----

## Sample Dataset

The application includes a built-in sample dataset for demonstration purposes. This dataset is a small CSV string containing CFD results for three different airfoil shapes (NACA 4412, NACA 0012, and Eppler 423) at various angles of attack.

| angle\_of\_attack | reynolds\_number | Cl | Cd | shape\_name |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 300000 | 0.45 | 0.007 | NACA 4412 |
| 2 | 300000 | 0.65 | 0.008 | NACA 4412 |
| 4 | 300000 | 0.85 | 0.012 | NACA 4412 |
| ... | ... | ... | ... | ... |

You can use this data by clicking the "Load Sample Data" button in the app's user interface. This is a great way to explore the app's functionality before uploading your own data.
