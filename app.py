import streamlit as st
import joblib
import pandas as pd

# 1. Load the trained Random Forest model
model = joblib.load("roof_fall_random_forest.pkl")

# 2. Page configuration
st.set_page_config(
    page_title="Roof Fall Risk Prediction",
    page_icon="⛏️",
    layout="centered"
)

# 3. Application title
st.title("⛏️ Underground Mine Roof Fall Risk Prediction")

st.write(
    "Enter the mine and roof-condition parameters below "
    "to estimate the roof fall risk."
)


# 4. Input variables
cmrr = st.number_input(
    "Coal Mine Roof Rating (CMRR)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

prsup = st.number_input(
    "Primary Roof Support (PRSUP)",
    min_value=0.0,
    value=5.0
)

intersection_span = st.number_input(
    "Intersection Span (ft)",
    min_value=0.0,
    value=60.0
)

depth = st.number_input(
    "Depth of Cover (ft)",
    min_value=0.0,
    value=500.0
)

mining_height = st.number_input(
    "Mining Height (ft)",
    min_value=0.0,
    value=7.0
)

# 5. Prediction
if st.button("Predict Roof Fall Risk"):

    # Create input dataframe in the same format
    # used while training the model

    input_data = pd.DataFrame(
        [[
            cmrr,
            prsup,
            intersection_span,
            depth,
            mining_height
        ]],
        columns=[
            "CMRR",
            "PRSUP",
            "Intersection_Span_ft",
            "Depth_of_Cover_ft",
            "Mining_Height_ft"
        ]
    )

    # Predict risk class
    prediction = model.predict(input_data)[0]

    # Predict class probabilities
    probabilities = model.predict_proba(input_data)[0]


    # Display Prediction

    st.divider()
    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("🟢 LOW ROOF FALL RISK")

    elif prediction == "Moderate":
        st.warning("🟡 MODERATE ROOF FALL RISK")

    elif prediction == "High":
        st.error("🔴 HIGH ROOF FALL RISK")


    # --------------------------------------------------
    # Risk Probabilities
    # --------------------------------------------------

    st.subheader("Risk Probabilities")

    probability_dict = {
        risk_class: probability
        for risk_class, probability in zip(
            model.classes_,
            probabilities
        )
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🟢 Low",
            f"{probability_dict.get('Low', 0) * 100:.1f}%"
        )

    with col2:
        st.metric(
            "🟡 Moderate",
            f"{probability_dict.get('Moderate', 0) * 100:.1f}%"
        )

    with col3:
        st.metric(
            "🔴 High",
            f"{probability_dict.get('High', 0) * 100:.1f}%"
        )


    # --------------------------------------------------
    # Probability Chart
    # --------------------------------------------------

    probability_df = pd.DataFrame({
        "Risk Level": ["Low", "Moderate", "High"],
        "Probability": [
            probability_dict.get("Low", 0),
            probability_dict.get("Moderate", 0),
            probability_dict.get("High", 0)
        ]
    })

    probability_df = probability_df.set_index("Risk Level")

    st.bar_chart(probability_df)