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


    # 6. Display prediction
 
    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("🟢 LOW ROOF FALL RISK")

    elif prediction == "Moderate":
        st.warning("🟡 MODERATE ROOF FALL RISK")

    elif prediction == "High":
        st.error("🔴 HIGH ROOF FALL RISK")


    # 7. Display probabilities
    
    st.subheader("Risk Probabilities")

    for risk_class, probability in zip(
        model.classes_,
        probabilities
    ):
        st.write(
            f"{risk_class}: {probability * 100:.2f}%"
        )


# 8. Disclaimer

st.divider()

st.caption(
    "This prediction system is developed for academic and "
    "research purposes. It should not replace statutory mine "
    "safety assessments or professional engineering judgement."
)