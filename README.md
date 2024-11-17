# MLCert

## Prerequisites

Ensure Python 3.8 or higher is installed.

Install the required dependencies:

```bash
pip install streamlit deepchecks
```

Run th3 command to download the military dataset:

```bash
curl -L "https://universe.roboflow.com/ds/7woZrwVehy?key=ejNWnZzaC1" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
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
