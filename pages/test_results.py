import streamlit as st
import plotly.graph_objects as go
import json
import pandas as pd
from PIL import Image
import io
import base64

WEIGHTS = {
    "high": {
        "Data Integrity": 35,
        "Train-Test Evaluation": 30,
        "Model Performance": 35,
    },
    "med": {"Data Integrity": 40, "Train-Test Evaluation": 30, "Model Performance": 30},
    "low": {"Data Integrity": 45, "Train-Test Evaluation": 30, "Model Performance": 25},
}


def get_thresholds_by_criticality(level):
    """Get mAP, Precision, and Recall thresholds by criticality level."""
    if level == "high":
        return 90, 90, 90  # mAP, Precision, Recall thresholds
    elif level == "med":
        return 85, 85, 85
    elif level == "low":
        return 80, 80, 80
    return 0, 0, 0  # Default


def check_pass_fail(value, threshold):
    """Check if a metric passes or fails based on the threshold."""
    return "PASS" if value >= threshold else "FAIL"


def calculate_final_score(
    data_integrity_score, train_test_score, model_perf_score, criticality_level="low"
):
    """Calculate the final weighted score based on the criticality level."""
    weights = WEIGHTS[criticality_level]
    final_score = (
        (data_integrity_score * weights["Data Integrity"] / 100)
        + (train_test_score * weights["Train-Test Evaluation"] / 100)
        + (model_perf_score * weights["Model Performance"] / 100)
    )
    return final_score


# Utility functions
def style_status(status):
    if status.lower() == "pass":
        return "color: #90EE90; font-weight: bold; padding: 10px; border-radius: 5px;"
    elif status.lower() == "fail":
        return "color: red; font-weight: bold; padding: 10px; border-radius: 5px;"
    return "color: gray; font-weight: bold; padding: 10px; border-radius: 5px;"


def status_icon(status):
    if status.lower() == "pass":
        return "✔️ PASSED"
    elif status.lower() == "fail":
        return "❌ FAILED"
    return "⚠️ Unknown"


def display_image_from_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image


def load_results():
    """Load the test results from a JSON file."""
    with open("outputs_sorted.json", "r") as f:
        return json.load(f)


def display_conditions(conditions):
    """Display the conditions from the result."""
    passed_conditions = 0
    for condition in conditions:
        st.markdown(
            f"<div style='{style_status(condition['Status'])}'>{status_icon(condition['Status'])}<br>"
            f"Condition: {condition['Condition']}<br>More Info: {condition['More Info']}</div>",
            unsafe_allow_html=True,
        )
        if condition["Status"] == "PASS":
            passed_conditions += 1
        elif condition["Status"] == "WARN":
            passed_conditions += 0.5
    return passed_conditions / len(conditions) if len(conditions) > 0 else 0


def display_drift_metrics(metrics):
    """Display drift metrics."""
    drift_df = pd.DataFrame(
        [
            {"Property": key, "Drift Score": val["Drift score"]["value"]}
            for key, val in metrics.items()
        ]
    )
    st.dataframe(drift_df)


def display_plotly_charts(displays):
    """Render plotly charts from the 'display' section."""
    for display in displays:
        if display["type"] == "plotly":
            plot_payload = json.loads(display["payload"])
            fig = go.Figure(plot_payload)
            st.plotly_chart(fig)


# Modify display functions to return final score
def display_data_integrity_section(results, outlier_threshold=10):
    """Display the Data Integrity section."""
    st.header("Data Integrity")

    def display_outliers(section_title, outliers_result):
        st.subheader(section_title)
        table_data = []
        total_score = 0
        for property_name, details in outliers_result["value"].items():
            outliers_identifiers = details.get("outliers_identifiers", [])
            if outliers_identifiers:
                if isinstance(outliers_identifiers["values"], str):
                    # If outliers_identifiers is a string, get its length using .shape
                    total_outliers = outliers_identifiers["shape"][0]
                else:
                    # If outliers_identifiers is a list or other iterable, use len()
                    total_outliers = len(outliers_identifiers["values"])

                lower_limit = round(details["lower_limit"]["value"], 2)
                upper_limit = round(details["upper_limit"]["value"], 2)
                table_data.append(
                    {
                        "Property": property_name,
                        "Total Outliers": total_outliers,
                        "Non-Outliers Range": f"{lower_limit} to {upper_limit}",
                    }
                )
                if total_outliers < outlier_threshold:
                    total_score += 1
            else:
                total_score += 1
        st.table(table_data)
        return total_score / len(outliers_result["value"])

    scores = []
    scores.append(
        display_outliers(
            "Image Property Outliers (Test Dataset)",
            results["image_property_outliers1"],
        )
    )
    scores.append(
        display_outliers(
            "Image Property Outliers (Train Dataset)",
            results["image_property_outliers2"],
        )
    )
    scores.append(
        display_outliers(
            "Label Property Outliers (Test Dataset)",
            results["label_property_outliers1"],
        )
    )
    scores.append(
        display_outliers(
            "Label Property Outliers (Train Dataset)",
            results["label_property_outliers2"],
        )
    )
    st.subheader("Property Label Correlation Change")
    scores.append(
        display_conditions(
            results["property_label_correlation_change"]["conditions_results"]
        )
    )

    total_checks = len(scores)
    total_score = sum(scores)
    final_score = round((total_score / total_checks), 2)

    st.markdown(f"### Overall Data Integrity Score: {final_score * 100:.2f}")
    return final_score


def display_train_test_validation_section(results):
    """Display the Train Test Validation section."""
    scores = []

    st.header("Train Test Validation")
    st.subheader("Image Property Drift")
    scores.append(
        display_conditions(results["image_property_drift"]["conditions_results"])
    )

    with st.expander("Image Property Drift Scores"):
        display_drift_metrics(results["image_property_drift"]["value"])

    with st.expander("Drift Plots"):
        display_plotly_charts(results["image_property_drift"]["display"])

    st.subheader("Label Drift")
    st.markdown(results["label_drift"]["check"]["summary"], unsafe_allow_html=True)
    scores.append(display_conditions(results["label_drift"]["conditions_results"]))

    with st.expander("Label Drift Metrics"):
        display_drift_metrics(results["label_drift"]["value"])

    with st.expander("Label Drift Plots"):
        display_plotly_charts(results["label_drift"]["display"])

    st.subheader("      ")
    st.markdown(
        results["property_label_correlation_change"]["check"]["summary"],
        unsafe_allow_html=True,
    )
    scores.append(
        display_conditions(
            results["property_label_correlation_change"]["conditions_results"]
        )
    )

    with st.expander("Property Label Correlation Change Metrics"):
        # Process metrics
        all_metrics = []
        for property_name, data in results["property_label_correlation_change"][
            "value"
        ].items():
            for category, value in data["train"].items():
                all_metrics.append(
                    {
                        "Property": property_name,
                        "Category": category,
                        "Train": value,
                        "Test": data["test"].get(category, 0),
                        "Difference": data["train-test difference"].get(category, 0),
                    }
                )
        df_metrics = pd.DataFrame(all_metrics)
        st.dataframe(df_metrics)

    with st.expander("Property Label Correlation Change Plot"):
        display_plotly_charts(results["property_label_correlation_change"]["display"])

    st.subheader("New Labels")
    st.markdown(results["new_labels"]["check"]["summary"], unsafe_allow_html=True)
    scores.append(display_conditions(results["new_labels"]["conditions_results"]))

    with st.expander("New Labels Metrics"):
        metrics = results["new_labels"]["value"]
        new_labels_df = pd.DataFrame(
            {
                "All Labels Count": [metrics["all_labels_count"]],
                "New Labels": [
                    (
                        ", ".join(metrics["new_labels"].keys())
                        if metrics["new_labels"]
                        else "None"
                    )
                ],
            }
        )
        st.dataframe(new_labels_df)

    total_checks = len(scores)
    total_score = sum(scores)
    final_score = round((total_score / total_checks), 2)

    st.markdown(f"### Overall Train Test Validation Score: {final_score * 100:.2f}")
    return final_score


def model_metrics(results=None):
    # Display the key metrics: mAP, Precision, and Recall
    st.subheader(f"Key Metrics (Criticality Level: low)")
    map_threshold, precision_threshold, recall_threshold = (
        get_thresholds_by_criticality("low")
    )
    map_score = 86.5
    precision_score = 82.1
    recall_score = 79.0
    # Check if each metric passes or fails based on thresholds
    map_status = check_pass_fail(map_score, map_threshold)
    precision_status = check_pass_fail(precision_score, precision_threshold)
    recall_status = check_pass_fail(recall_score, recall_threshold)

    metrics_cond = []
    metrics_cond.append(
        {
            "Status": map_status,
            "Condition": "mAP Score > 80",
            "More Info": "Mean Average Precision",
        }
    )
    metrics_cond.append(
        {
            "Status": precision_status,
            "Condition": "Precision Score > 80",
            "More Info": "Precision",
        }
    )
    metrics_cond.append(
        {
            "Status": recall_status,
            "Condition": "Recall Score > 80",
            "More Info": "Recall",
        }
    )
    display_conditions(metrics_cond)
    # Display the metrics as a table with pass/fail status
    key_metrics_df = pd.DataFrame(
        {
            "Metric": ["mAP", "Precision", "Recall"],
            "Score ()": [map_score, precision_score, recall_score],
            "Threshold ()": [map_threshold, precision_threshold, recall_threshold],
            "Status": [map_status, precision_status, recall_status],
        }
    )
    st.table(key_metrics_df)

    score = 0
    for i in metrics_cond:
        if i["Status"] == "PASS":
            score += 1
    return score / len(metrics_cond)


def display_model_evaluation_section(results):
    """Display the Model Evaluation section."""
    st.header("Model Evaluation")
    metrics_score = model_metrics()
    st.subheader("Class Performance Evaluation")
    scores = []
    scores.append(metrics_score)
    st.markdown(
        results["class_performance"]["check"]["summary"], unsafe_allow_html=True
    )
    scores.append(
        display_conditions(results["class_performance"]["conditions_results"])
    )

    with st.expander("Class Performance Metrics"):
        df_metrics = pd.DataFrame(json.loads(results["class_performance"]["value"]))
        st.dataframe(df_metrics)

    with st.expander("Class Performance Plot"):
        display_plotly_charts(results["class_performance"]["display"])

    st.subheader("Prediction Drift")
    st.markdown(results["prediction_drift"]["check"]["summary"], unsafe_allow_html=True)
    scores.append(display_conditions(results["prediction_drift"]["conditions_results"]))

    with st.expander("Prediction Drift Metrics"):
        display_drift_metrics(results["prediction_drift"]["value"])

    with st.expander("Prediction Drift Plot"):
        display_plotly_charts(results["prediction_drift"]["display"])
    total_checks = len(scores)
    total_score = sum(scores)
    final_score = round((total_score / total_checks), 2)

    st.markdown(f"### Overall Model Evaluation Score: {final_score * 100:.2f}")
    return final_score


# Main page function with table added at the top
def test_results_page():
    """Main function to render the test results page."""
    st.title("Test Results")

    # Load deepchecks result
    deepchecks_result = load_results()

    # Extract relevant sections from results
    results = {
        "class_performance": deepchecks_result["results"][0],
        "image_property_drift": deepchecks_result["results"][5],
        "prediction_drift": deepchecks_result["results"][16],
        "label_drift": deepchecks_result["results"][8],
        "property_label_correlation_change": deepchecks_result["results"][19],
        "new_labels": deepchecks_result["results"][15],
        "image_property_outliers1": deepchecks_result["results"][6],
        "image_property_outliers2": deepchecks_result["results"][7],
        "label_property_outliers1": deepchecks_result["results"][9],
        "label_property_outliers2": deepchecks_result["results"][10],
    }

    # Calculate scores for each section
    data_integrity_score = display_data_integrity_section(
        results, outlier_threshold=250
    )
    train_test_validation_score = display_train_test_validation_section(results)
    model_evaluation_score = display_model_evaluation_section(results)

    # Display score table at the top
    st.subheader("Overall Scores Summary")
    scores_summary_df = pd.DataFrame(
        {
            "Section": ["Data Integrity", "Train Test Validation", "Model Evaluation"],
            "Score": [
                f"{data_integrity_score * 100:.2f}",
                f"{train_test_validation_score * 100:.2f}",
                f"{model_evaluation_score * 100:.2f}",
            ],
        }
    )

    # criticality_level = st.session_state["inputs"]["criticality_level"]
    criticality_level = "low"

    final_score = calculate_final_score(
        data_integrity_score,
        train_test_validation_score,
        model_evaluation_score,
        criticality_level,
    )
    st.table(scores_summary_df)


def display_nutrition_table():
    """Display the model nutrition table with calculated overall scores."""

    # Subpart Scores
    data_integrity_scores = [80, 85, 90, 87, 82]
    train_test_validation_scores = [78, 75, 82, 85]
    model_evaluation_scores = [95, 89, 88, 92]

    # Calculate overall scores as averages of subparts
    data_integrity_avg = round(
        sum(data_integrity_scores) / len(data_integrity_scores), 2
    )
    train_test_validation_avg = round(
        sum(train_test_validation_scores) / len(train_test_validation_scores), 2
    )
    model_evaluation_avg = round(
        sum(model_evaluation_scores) / len(model_evaluation_scores), 2
    )
    weights = WEIGHTS["low"]

    # Weighted score calculation
    final_score = (
        (data_integrity_avg * weights["Data Integrity"])
        + (train_test_validation_avg * weights["Train-Test Evaluation"])
        + (model_evaluation_avg * weights["Model Performance"])
    ) / 100  # Divide by 100 to normalize percentages
    st.markdown(
        f"<h3 style='color:#90EE90;'>Final Automated Score (Low Criticality): {final_score:.2f}%</h3>",
        unsafe_allow_html=True,
    )
    # Data Integrity Section
    st.subheader(f"Data Integrity - {data_integrity_avg}")
    st.markdown(
        """
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <thead >
                <tr>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Subpart</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Image Property Outliers (Test Dataset)</td>
                    <td style="border:1px solid #ddd; padding:8px;">80</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Image Property Outliers (Train Dataset)</td>
                    <td style="border:1px solid #ddd; padding:8px;">85</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Label Property Outliers (Test Dataset)</td>
                    <td style="border:1px solid #ddd; padding:8px;">90</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Label Property Outliers (Train Dataset)</td>
                    <td style="border:1px solid #ddd; padding:8px;">87</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Property-Label Correlation Change</td>
                    <td style="border:1px solid #ddd; padding:8px;">82</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Train-Test Validation Section
    st.subheader(f"Train-Test Validation - {train_test_validation_avg}")
    st.markdown(
        """
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <thead>
                <tr>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Subpart</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Image Property Drift</td>
                    <td style="border:1px solid #ddd; padding:8px;">78</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Label Drift</td>
                    <td style="border:1px solid #ddd; padding:8px;">75</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Property-Label Correlation Change</td>
                    <td style="border:1px solid #ddd; padding:8px;">82</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">New Labels</td>
                    <td style="border:1px solid #ddd; padding:8px;">85</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Model Evaluation Section
    st.subheader(f"Model Evaluation - {model_evaluation_avg}")
    st.markdown(
        """
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <thead>
                <tr>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Subpart</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Key Metrics Score</td>
                    <td style="border:1px solid #ddd; padding:8px;">95</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Class Performance</td>
                    <td style="border:1px solid #ddd; padding:8px;">89</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Prediction Drift</td>
                    <td style="border:1px solid #ddd; padding:8px;">88</td>
                </tr>
                <tr>
                    <td style="border:1px solid #ddd; padding:8px;">Inference Time</td>
                    <td style="border:1px solid #ddd; padding:8px;">92</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":

    st.header("Evaluation Summary")

    display_nutrition_table()

    test_results_page()
