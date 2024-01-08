import streamlit as st
from joblib import load

# Paths to the saved models
svm_model_path = 'OneDrive - Solent University/COM726_Dissertation_Project/Project Supervisor/Epileptic Seizure/svm_esr_model.joblib'
rf_model_path = 'OneDrive - Solent University/COM726_Dissertation_Project/Project Supervisor/Epileptic Seizure/rf_esr_model.joblib'
dt_model_path = 'OneDrive - Solent University/COM726_Dissertation_Project/Project Supervisor/Epileptic Seizure/dt_esr_model.joblib'
knn_model_path = 'OneDrive - Solent University/COM726_Dissertation_Project/Project Supervisor/Epileptic Seizure/knn_esr_model.joblib'

# Load saved models using joblib
def load_model(model_path):
    try:
        model = load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        return None

# Function for making predictions with the loaded models
def predict_with_model(model, input_data):
    if model:
        # Perform prediction using the loaded model
        prediction = model.predict(input_data)
        return prediction
    else:
        return None

def home():
    st.title("Epileptic Seizures Prediction")
    st.write("Welcome to the Epileptic Seizures Prediction App!")
    
    st.header("What are Epileptic Seizures?")
    st.write("Epileptic seizures are sudden, uncontrolled electrical disturbances in the brain. These disruptions can cause changes in behavior, sensations, and consciousness. Seizures vary in type, duration, and severity, and can be triggered by various factors.")
    
    st.header("Types of Seizures")
    st.write("There are different types of seizures, including generalized seizures (affecting both sides of the brain) and focal seizures (occurring in one area of the brain). Each type can manifest differently and may involve various symptoms.")
    
    st.header("Treatment and Management")
    st.write("Treatment for epilepsy and seizures often involves medications to control seizures, lifestyle modifications, and in some cases, surgery. It's important for individuals with epilepsy to work closely with healthcare professionals to manage their condition.")

def predict_seizure_page():
    st.title("Seizure Prediction")
    st.write("Enter input features for seizure prediction:")

    # Simulated input fields for features
    feature_1 = st.number_input("Feature 1")
    feature_2 = st.number_input("Feature 2")

    # Prepare input data for prediction
    input_data = [[feature_1, feature_2]]  # Adapt this based on your model input requirements

    # Sidebar navigation for model selection
    selected_model = st.sidebar.selectbox("Select Model", ('SVM', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors'))

    # Load selected model
    if selected_model == 'SVM':
        model = load_model(svm_model_path)
    elif selected_model == 'Random Forest':
        model = load_model(rf_model_path)
    elif selected_model == 'Decision Tree':
        model = load_model(dt_model_path)
    elif selected_model == 'K-Nearest Neighbors':
        model = load_model(knn_model_path)
    else:
        model = None

    # Perform prediction using the selected model
    if model:
        # Make prediction based on the selected model
        prediction_result = predict_with_model(model, input_data)
        if prediction_result is not None:
            st.write(f"{selected_model} Prediction:", prediction_result)
        else:
            st.warning("Please provide valid input data for prediction.")
    else:
        st.warning("Please select a model.")

def about_epilepsy():
    st.title("About Epileptic Seizures")
    # Add content for the about page

def main():
    # Sidebar navigation
    menu = st.sidebar.selectbox(
        'Navigation',
        ('Home', 'Prediction', 'About Epileptic Seizures')
    )

    # Display selected page based on menu choice
    if menu == 'Home':
        home()
    elif menu == 'Prediction':
        predict_seizure_page()
    elif menu == 'About Epileptic Seizures':
        about_epilepsy()

if __name__ == "__main__":
    main()

