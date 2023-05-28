import requests
import streamlit as st
import hydralit_components as hc

# Define the input fields
inputs_form = [
    ("race", "Race"),
    ("gender", "Gender"),
    ("age", "Age"),
    ("discharge_disposition_id", "Discharge Disposition ID"),
    ("admission_source_id", "Admission Source ID"),
    ("time_in_hospital", "Time in Hospital"),
    ("num_lab_procedures", "Number of Lab Procedures"),
    ("num_procedures", "Number of Procedures"),
    ("num_medications", "Number of Medications"),
    ("number_outpatient", "Number of Outpatient Visits"),
    ("number_emergency", "Number of Emergency Visits"),
    ("number_inpatient", "Number of Inpatient Visits"),
    ("diag_1", "Diagnosis 1"),
    ("diag_2", "Diagnosis 2"),
    ("diag_3", "Diagnosis 3"),
    ("number_diagnoses", "Number of Diagnoses"),
    ("max_glu_serum", "Max Glucose Serum"),
    ("A1Cresult", "A1C Result"),
    ("metformin", "Metformin"),
    ("repaglinide", "Repaglinide"),
    ("nateglinide", "Nateglinide"),
    ("chlorpropamide", "Chlorpropamide"),
    ("glimepiride", "Glimepiride"),
    ("acetohexamide", "Acetohexamide"),
    ("glipizide", "Glipizide"),
    ("glyburide", "Glyburide"),
    ("tolbutamide", "Tolbutamide"),
    ("pioglitazone", "Pioglitazone"),
    ("rosiglitazone", "Rosiglitazone"),
    ("acarbose", "Acarbose"),
    ("miglitol", "Miglitol"),
    ("troglitazone", "Troglitazone"),
    ("tolazamide", "Tolazamide"),
    ("examide", "Examide"),
    ("citoglipton", "Citoglipton"),
    ("insulin", "Insulin"),
    ("glyburide_metformin", "Glyburide Metformin"),
    ("glipizide_metformin", "Glipizide Metformin"),
    ("glimepiride_pioglitazone", "Glimepiride Pioglitazone"),
    ("metformin_rosiglitazone", "Metformin Rosiglitazone"),
    ("metformin_pioglitazone", "Metformin Pioglitazone"),
    ("change", "Change"),
    ("diabetesMed", "Diabetes Medication"),
]


# Set Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom styles to the form elements
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #FF5722;
        color: white;
        font-weight: bold;
        padding: 0.25rem 0.5rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stTextInput > label {
    font-size:120%;
    font-weight:bold;
    color:black;
    background:linear-gradient(to bottom, #3399ff 0%,##F3F3F3 100%);
    border: 2px;
    border-radius: 3px;
    }

    [data-baseweb="input"]{
    background:linear-gradient(to bottom, #3399ff 0%,##F3F3F3 100%);
    border: 2px;
    border-radius: 3px;
     width: 50%;
    }

    input[class]{
    font-weight: bold;
    font-size:120%;
    color: black;
    }
    .small-input {
        max-width: 50px;
    }
    .column {
        float: left;
        width: 15%;
        padding: 0 10px;
    }
    .row::after {
        content: "";
        clear: both;
        display: table;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a form for input values
def create_input_form():
    inputs = {}

    default_values = { "race": "Caucasian",
                        "gender": "Female",
                        "age": "50-60",
                        "discharge_disposition_id": 1,
                        "admission_source_id": 7,
                        "time_in_hospital": 4,
                        "num_lab_procedures": 40,
                        "num_procedures": 2,
                        "num_medications": 12,
                        "number_outpatient": 0,
                        "number_emergency": 1,
                        "number_inpatient": 0,
                        "diag_1": "428",
                        "diag_2": "250",
                        "diag_3": "401",
                        "number_diagnoses": 6,
                        "max_glu_serum": "None",
                        "A1Cresult": "None",
                        "metformin": "No",
                        "repaglinide": "No",
                        "nateglinide": "No",
                        "chlorpropamide": "No",
                        "glimepiride": "No",
                        "acetohexamide": "No",
                        "glipizide": "No",
                        "glyburide": "No",
                        "tolbutamide": "No",
                        "pioglitazone": "No",
                        "rosiglitazone": "No",
                        "acarbose": "No",
                        "miglitol": "No",
                        "troglitazone": "No",
                        "tolazamide": "No",
                        "examide": "No",
                        "citoglipton": "No",
                        "insulin": "No",
                        "glyburide_metformin": "No",
                        "glipizide_metformin": "No",
                        "glimepiride_pioglitazone": "No",
                        "metformin_rosiglitazone": "No",
                        "metformin_pioglitazone": "No",
                        "change": "No",
                        "diabetesMed": "Yes"
                        }

    with st.form("Diabetes Form"):
        col1, col2, col3,col4,col5,col6 = st.columns(6)  # Split the form into three columns

        # Create input fields in the first column
        with col1:
            for i in range(len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        # Create input fields in the second column
        with col2:
            for i in range(len(inputs_form) // 6, 2 * len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        # Create input fields in the third column
        with col3:
            for i in range(2 * len(inputs_form) // 6, 3 * len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        with col4:
            for i in range(3 * len(inputs_form) // 6, 4 * len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        # Create input fields in the second column
        with col5:
            for i in range(4 * len(inputs_form) // 6, 5 * len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        # Create input fields in the third column
        with col6:
            for i in range(5 * len(inputs_form) // 6, 6 * len(inputs_form) // 6):
                attribute, label = inputs_form[i]
                value = st.text_input(label, key=attribute, value=default_values[attribute])
                inputs[attribute] = value

        # Apply custom styles to the submit button
        st.markdown(
            """
            <style>
            .stButton button:hover {
                background-color: #E64A19;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        submit_button = st.form_submit_button("Predict")

    return inputs, submit_button

# Main function
def main():
    st.title("Diabetes Prediction - Proyecto Final")

    with hc.HyLoader('Predicting...',hc.Loaders.standard_loaders,index=[0]):
     inputs, submit_button = create_input_form()
     if submit_button:
        if any(value == "" for value in inputs.values()):
            st.error("Error: Please fill in all the input fields.")
        else:
            # Prepare the data payload
            data = inputs
            # Make a POST request to  FastAPI endpoint
            
            response = requests.post("http://10.43.102.111:8503/do_inference/diabetes", json=data)
    
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                st.success(result)
            else:
                st.error("Error: Failed to make the prediction.")

# Run the application
if __name__ == "__main__":
    main()