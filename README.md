# Automated Checks for Certification Approach

## Prerequisites

Ensure Python 3.8 or higher is installed.

Install the required dependencies:

```bash
pip install streamlit deepchecks
```

Run the command to download the military dataset:

```bash
curl -L "https://universe.roboflow.com/ds/O79Fya4DXb?key=rVXaAECSDB" > Military\ vehicles\ object\ detection.v16i.coco.zip
```

## How to Run

1. Start the streamlit app

```bash
streamlit run app.py --server.maxUploadSize 300
```

2. Upload the required files:

   Model file: best.onnx
   Dataset: Military vehicles object detection.v16i.coco.zip

3. Run the automated checks:

   Use the app interface to execute the tests on the model and dataset.

4. View the results:

   Go to the Test Results page to see metrics, scores, and visual plots.
