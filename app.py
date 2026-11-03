
import streamlit as st
import joblib
import pandas as pd

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Mine Roof Fall Risk Prediction",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("roof_fall_random_forest.pkl")


model = load_model()

MODEL_FEATURES = [
    "CMRR",
    "PRSUP",
    "Intersection_Span_ft",
    "Depth_of_Cover_ft",
    "Mining_Height_ft"
]

RISK_ORDER = ["Low", "Moderate", "High"]


# ==========================================================
# 3. SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("⛏️ MI302 Project")

    st.markdown("---")

    st.subheader("About the System")

    st.write(
        """
        This application uses a trained Random Forest
        classification model to estimate roof fall risk
        in underground mines.

        Five mining parameters are used as model inputs.
        """
    )

    st.markdown("---")

    st.subheader("Risk Classification")

    st.markdown("🟢 **Low Risk**")
    st.markdown("🟡 **Moderate Risk**")
    st.markdown("🔴 **High Risk**")

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Input Features:** 5")
    st.write("**Dataset Size:** 109 observations")

    st.markdown("---")

    st.caption(
        "MI302 – Mine Hazards, Rescue and Recovery"
    )


# ==========================================================
# 4. MAIN HEADER
# ==========================================================

st.title("⛏️ Underground Mine Roof Fall Risk Prediction")

st.write(
    """
    Estimate roof fall risk using mine geometry and
    roof-condition parameters through a trained
    Machine Learning model.
    """
)

st.info(
    "Enter the mining parameters below and click "
    "**Predict Roof Fall Risk** to obtain the prediction."
)


# ==========================================================
# 5. INPUT PARAMETERS
# ==========================================================

st.header("Mine and Roof Parameters")

st.caption(
    "Input limits represent the observed minimum and maximum "
    "values in the 109-observation training dataset. "
    "They are not statutory safety limits."
)

# ----------------------------------------------------------
# Roof Condition
# ----------------------------------------------------------

st.subheader("1. Roof Condition")

col1, col2 = st.columns(2)

with col1:

    cmrr = st.number_input(
        "Coal Mine Roof Rating (CMRR)",
        min_value=28.0,
        max_value=78.0,
        value=50.0,
        step=1.0,
        help=(
            "Rating representing the structural quality "
            "of the coal mine roof."
        )
    )

    st.caption("Observed dataset range: 28–78")

with col2:

    prsup = st.number_input(
        "Primary Roof Support (PRSUP)",
        min_value=2.46,
        max_value=14.67,
        value=5.0,
        step=0.1,
        help=(
            "Primary roof support parameter as represented "
            "in the training dataset."
        )
    )

    st.caption("Observed dataset range: 2.46–14.67")


# ----------------------------------------------------------
# Mine Geometry
# ----------------------------------------------------------

st.subheader("2. Mine Geometry")

col1, col2 = st.columns(2)

with col1:

    intersection_span = st.number_input(
        "Intersection Span (ft)",
        min_value=50.0,
        max_value=78.4,
        value=60.0,
        step=0.1,
        help="Intersection span measured in feet."
    )

    st.caption("Observed dataset range: 50–78.4 ft")

    depth = st.number_input(
        "Depth of Cover (ft)",
        min_value=150.0,
        max_value=1100.0,
        value=500.0,
        step=10.0,
        help=(
            "Depth of overburden above the mining horizon."
        )
    )

    st.caption("Observed dataset range: 150–1100 ft")

with col2:

    mining_height = st.number_input(
        "Mining Height (ft)",
        min_value=3.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="Height of the mined opening in feet."
    )

    st.caption("Observed dataset range: 3–10 ft")


# ==========================================================
# 6. PREDICTION
# ==========================================================

st.divider()

if st.button(
    "🔍 Predict Roof Fall Risk",
    type="primary",
    use_container_width=True
):

    # Create input DataFrame in training feature order

    input_data = pd.DataFrame(
        [[
            cmrr,
            prsup,
            intersection_span,
            depth,
            mining_height
        ]],
        columns=MODEL_FEATURES
    )

    # Model prediction

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    # Map probabilities to their respective classes

    probability_dict = {
        str(risk_class): float(probability)
        for risk_class, probability in zip(
            model.classes_,
            probabilities
        )
    }

    # ------------------------------------------------------
    # Prediction Result
    # ------------------------------------------------------

    st.header("Prediction Result")

    if prediction == "Low":

        st.success("🟢 LOW ROOF FALL RISK")

        st.write(
            "The model classifies the entered conditions "
            "as Low Risk."
        )

    elif prediction == "Moderate":

        st.warning("🟡 MODERATE ROOF FALL RISK")

        st.write(
            "The model classifies the entered conditions "
            "as Moderate Risk."
        )

    elif prediction == "High":

        st.error("🔴 HIGH ROOF FALL RISK")

        st.write(
            "The model classifies the entered conditions "
            "as High Risk."
        )

    else:

        st.write(f"Predicted class: {prediction}")

    # ------------------------------------------------------
    # Risk Probabilities
    # ------------------------------------------------------

    st.subheader("Risk Probabilities")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🟢 Low Risk",
            f"{probability_dict.get('Low', 0) * 100:.2f}%"
        )

    with col2:

        st.metric(
            "🟡 Moderate Risk",
            f"{probability_dict.get('Moderate', 0) * 100:.2f}%"
        )

    with col3:

        st.metric(
            "🔴 High Risk",
            f"{probability_dict.get('High', 0) * 100:.2f}%"
        )

    # ------------------------------------------------------
    # Probability Chart
    # ------------------------------------------------------

    st.subheader("Probability Distribution")

    probability_df = pd.DataFrame({
        "Risk Level": RISK_ORDER,
        "Probability": [
            probability_dict.get(risk, 0)
            for risk in RISK_ORDER
        ]
    })

    st.bar_chart(
        probability_df,
        x="Risk Level",
        y="Probability"
    )

    # ------------------------------------------------------
    # Input Summary
    # ------------------------------------------------------

    with st.expander("View Entered Mine Parameters"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ==========================================================
# 7. MODEL INSIGHTS
# ==========================================================

st.divider()

st.header("📊 Model Insights")

st.subheader("Random Forest Feature Importance")

st.write(
    """
    Feature importance indicates how much each input
    feature contributes to impurity reduction across
    the fitted Random Forest trees.
    """
)

# Extract feature importance directly from trained model

importance_df = pd.DataFrame({
    "Feature": MODEL_FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

importance_df["Feature"] = importance_df["Feature"].replace({
    "Intersection_Span_ft": "Intersection Span",
    "Depth_of_Cover_ft": "Depth of Cover",
    "Mining_Height_ft": "Mining Height"
})

importance_df["Importance_Percentage"] = (
    importance_df["Importance"] * 100
)

col1, col2 = st.columns([3, 1])

with col1:

    st.bar_chart(
        importance_df,
        x="Feature",
        y="Importance_Percentage",
        horizontal=True
    )

with col2:

    st.write("**Importance (%)**")

    for _, row in importance_df.sort_values(
        "Importance_Percentage",
        ascending=False
    ).iterrows():

        st.metric(
            row["Feature"],
            f"{row['Importance_Percentage']:.2f}%"
        )

st.caption(
    "Feature importance is model-specific and does not "
    "establish causal relationships between variables "
    "and roof falls."
)


# ==========================================================
# 8. MODEL PERFORMANCE
# ==========================================================

st.divider()

st.header("📈 Model Performance")

st.write(
    """
    The following results are taken from the tuned
    Random Forest evaluation in the project notebook.
    """
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Accuracy",
        "54.55%"
    )

with col2:

    st.metric(
        "Test Macro F1-Score",
        "0.528"
    )

with col3:

    st.metric(
        "Training Accuracy",
        "86.21%"
    )

st.caption(
    "These are the reported holdout test and training "
    "results from the notebook, not a live evaluation."
)

st.warning(
    """
    The current test accuracy is approximately 54.55%.
    The model requires further validation and improvement
    before any practical mine-safety application.
    """
)


# ==========================================================
# 9. PROJECT INFORMATION
# ==========================================================

st.divider()

st.header("About the Project")

st.write(
    """
    **Project:** Underground Mine Roof Fall Risk Prediction

    **Course:** MI302 – Mine Hazards, Rescue and Recovery

    **Model:** Tuned Random Forest Classifier

    **Objective:** To explore the use of Machine Learning
    for classifying roof fall risk based on selected
    mining and roof-condition parameters.
    """
)


# ==========================================================
# 10. DISCLAIMER
# ==========================================================

st.divider()

st.subheader("⚠️ Important Disclaimer")

st.warning(
    """
    This application is developed for academic and
    research purposes only.

    The predicted risk and class probabilities are
    outputs of a machine learning model trained on a
    limited dataset. They should not be interpreted as
    calibrated probabilities of an actual roof fall.

    This system must not replace statutory mine-safety
    assessments, geotechnical investigations, approved
    roof-support plans, or professional engineering
    judgement.
    """
)

st.caption(
    "MI302 Academic Project | Underground Mine Roof Fall "
    "Risk Prediction System"
)
