import streamlit as st
import zipfile
import tempfile
from ultralytics import YOLO
from checks import main

inputs = {}


# Function to extract dataset zip file
def process_dataset_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


# Function to run Deepchecks
def run_deepchecks(dataset_path, model):

    # training_data, test_data = main(model, dataset_path)
    # st.session_state["training_data"] = training_data
    # st.session_state["test_data"] = test_data
    # print(st.session_state["training_data"])
    st.session_state["deepchecks_result"] = main(model, dataset_path)


# Main page function
def main_page():
    # st.set_page_config(
    #     page_title=None,
    #     page_icon=None,
    #     layout="centered",
    #     initial_sidebar_state="collapsed",
    # )
    st.title("Model & Dataset Validation")

    uploaded_onnx = st.file_uploader("Upload your model (.onnx file)", type="onnx")
    uploaded_zip = st.file_uploader("Upload your dataset (.zip file)", type="zip")

    if uploaded_onnx and uploaded_zip:
        st.write("Processing the uploads...")

        # Save the uploaded ONNX model file
        temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
        temp_model_file.write(uploaded_onnx.read())
        temp_model_file.close()

        # Load model
        model = YOLO(temp_model_file.name, task="detect")
        st.write(f"ONNX model loaded from: {uploaded_onnx.name}")

        # Process dataset
        dataset_path = process_dataset_zip(uploaded_zip)
        st.write(f"Dataset extracted to: {dataset_path}")

        inputs["model"] = model
        inputs["dataset_path"] = dataset_path
        inputs["criticality_level"] = "low"

        st.session_state["inputs"] = inputs

        if st.button("Calculate Scores"):
            st.write("Running Checks...")
            run_deepchecks(dataset_path, model)
            st.success(
                "Scores calculated! View the results in the 'Test Results' page."
            )

    else:
        st.warning("Please upload both the model (.onnx) and dataset (.zip) files.")


# Main function to launch the app
if __name__ == "__main__":
    main_page()
