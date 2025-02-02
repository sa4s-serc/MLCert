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

# **Air Sight Certification Scoring Methodology**

## **1. Activity Score Determination**

The certification process consists of four main processes: Development, Verification & Validation (V&V), Quality Assurance (QA), and Configuration Management (CM). Each process includes multiple activities evaluated through manual and automated checks.

Each activity score is determined based on:

- **Manual Review Scores**: Evaluation of documentation, integration, usability, and compliance.
- **Automated Test Scores**: Performance metrics, robustness testing, and dataset validation.
- **Pass Rate of Automated Tests**: The percentage of successful automated checks.
- **Operational Relevance**: The impact of the activity on system safety and performance.

The final activity score is calculated as:

```math
S_{act_i} = \frac{\text{Manual Score} + \text{Automated Score}}{2}
```

---

## **2. Process Weights and Activity Breakdown**

### **Development Process (Weight = 0.30)**

The **30% weight** reflects the foundational role of development in ML certification, ensuring high-quality data and model documentation.

#### **Dataset Quality (Weight = 0.40)**

- **Highest weight in Development** due to ML's reliance on data.
- Score components:
  - **Data Completeness (30%)** – Coverage of all operational scenarios.
  - **Label Accuracy (40%)** – Precision of annotations.
  - **Class Balance (30%)** – Fair representation of vehicle types.

#### **Model Documentation (Weight = 0.35)**

- Ensures traceability and explainability.
- Score components:
  - **Architecture Documentation (40%)** – YOLOv8 structure, layers.
  - **Training Process Documentation (35%)** – Hyperparameters, optimization.
  - **Performance Specifications (25%)** – Expected precision, recall, inference speed.

#### **Integration Documentation (Weight = 0.25)**

- Lower weight due to reduced integration complexity in Level D systems.
- Score components:
  - **Interface Specifications (40%)** – API, ML & non-ML communication.
  - **Data Flow Documentation (35%)** – System pipeline.
  - **Error Handling Procedures (25%)** – Fault tolerance, exceptions.

---

### **Verification & Validation Process (Weight = 0.35)**

The **highest weighted process (35%)**, ensuring system reliability and compliance.

#### **Model Performance & Robustness Testing (Weight = 0.25 each)**

- **Highest weights** due to direct impact on operations.
- Performance Scoring:
  - **Accuracy Metrics (40%)** – Precision, recall, mAP.
  - **Real-time Performance (35%)** – Inference speed.
  - **Resource Utilization (25%)** – CPU, memory efficiency.

#### **Dataset Certification (Weight = 0.20)**

- Verifies dataset integrity and relevance.
- Score components:
  - **Distribution Analysis (40%)** – Identifying biases, gaps.
  - **Drift Detection (35%)** – Monitoring dataset evolution.
  - **Anomaly Identification (25%)** – Detecting errors.

#### **System Integration & Human Factors (Weight = 0.15 each)**

- **Lower weight** but essential for usability and operational trust.
- **System Integration Scoring:**
  - **Interface Testing (50%)** – ML & non-ML communication.
  - **End-to-End Validation (50%)** – System-wide verification.
- **Human Factors Scoring:**
  - **Trustworthiness (40%)** – Operator confidence in AI.
  - **Usability (35%)** – System responsiveness.
  - **Cognitive Load Reduction (25%)** – Ease of use.

---

### **Quality Assurance Process (Weight = 0.20)**

QA ensures compliance and long-term monitoring, assigned **20% weight**.

#### **Post-Certification Monitoring & Usability (Weight = 0.35 each)**

- **Highest QA weights** since monitoring ensures sustained reliability.
- **Monitoring Scoring:**
  - **Performance Tracking (40%)** – Continuous evaluation.
  - **Drift Monitoring (35%)** – Dataset/model behavior.
  - **Issue Resolution (25%)** – Speed of problem resolution.

#### **Audits and Reviews (Weight = 0.30)**

- Score components:
  - **Documentation Completeness (40%)** – Traceability.
  - **Process Adherence (35%)** – Compliance verification.
  - **Change Management (25%)** – Software update handling.

---

### **Configuration Management Process (Weight = 0.15)**

Lowest weight due to **limited update frequency** for Level D systems.

#### **Version Control (Weight = 0.40)**

- Most critical in CM for tracking system changes.
- Score components:
  - **Model Versioning (40%)** – YOLOv8 iterations.
  - **Dataset Versioning (35%)** – Dataset modifications.
  - **Documentation Updates (25%)** – Log maintenance.

---

## **3. Score Calculation Methodology**

Each activity score is normalized to 100 points:

```math
S_{act} = \left( \frac{\text{Achieved Points}}{\text{Total Possible Points}} \right) \times 100
```

The **weighted process score** is calculated as:

```math
S_{process} = \sum w_{act} \cdot S_{act}
```

The **final certification score** is determined as:

```math
S_{cert} = \sum w_{process} \cdot S_{process}
```

---

## **4. Certification Level Determination**

The final certification score maps to the following confidence levels:

| **Certification Score (S_cert)** | **Confidence Level**   |
| -------------------------------- | ---------------------- |
| 90 - 100                         | Optimal Assurance      |
| 80 - 89                          | Strong Assurance       |
| 70 - 79                          | Moderate Assurance     |
| 60 - 69                          | Limited Assurance      |
| < 60                             | Insufficient Assurance |

For **Air Sight**, with **`S_cert = 74.7`**, the system achieves **Moderate Assurance**, indicating compliance with **Level D requirements** with recommendations for improvements in QA and CM.

---

### **Process-Level Weights**

- **V&V (35%)** – Primary safety and reliability assurance
- **Development (30%)** – Foundation for ML system quality.
- **QA (20%)** – Ongoing compliance monitoring.
- **CM (15%)** – Static nature of Level D models.

Note: Some scores (e.g., model performance, dataset quality) are based on actual calculations. However, aspects such as Configuration Management, Monitoring Process, and parts of the manual reviews (especially documentation-related scores) contain example or estimated values. Future iterations will refine these based on empirical assessments and operational feedback.
