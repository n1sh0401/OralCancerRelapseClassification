import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="OSCC Risk Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("oscc_LR_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor


model, preprocessor = load_artifacts()

FEATURE_COLUMNS = [
    "AGE",
    "SEX",
    "ETHNICITY",
    "ALCOHOL",
    "SMOKER",
    "BETEL NUT CHEWER",
    "COMORBID",
    "TUMOUR SIZE(pT)",
    "NODAL INVOLVEMENT",
    "ENE",
    "p(N)",
    "pTNM STAGING",
    "VARIANT",
    "DOI(mm)",
    "TUMOUR DIMENSION(mm)",
    "wPOI",
    "LHR",
    "BONE INVASION",
    "BONE INVASION (TYPE)",
    "SURGICAL MARGIN",
    "DYSPLASIA AT SURGICAL MARGIN",
    "DYSPLASIA AT SURGICAL MARGIN (GRADING)",
    "TUMOUR GRADING",
    "PNI",
    "LVI",
    "PRIMARY TUMOUR SITE",
    "TYPE OF PRIMARY SURGICAL INTERVENTION",
    "POST ADJUVANT THERAPY",
    "RECURRENCE (LR, LRR)",
]

DEFAULT_VALUES = {
    "AGE": 50,
    "SEX": "MALE",
    "ETHNICITY": "INDIAN",
    "ALCOHOL": "NO",
    "SMOKER": "NO",
    "BETEL NUT CHEWER": "NO",
    "COMORBID": "NO",
    "TUMOUR SIZE(pT)": 2,
    "NODAL INVOLVEMENT": "NO",
    "ENE": "NO",
    "p(N)": 0,
    "pTNM STAGING": "II",
    "VARIANT": "CONVENTIONAL",
    "DOI(mm)": 7,
    "TUMOUR DIMENSION(mm)": 10,
    "wPOI": 3,
    "LHR": "STRONG",
    "BONE INVASION": "NO",
    "BONE INVASION (TYPE)": "ND",
    "SURGICAL MARGIN": "CLEAR",
    "DYSPLASIA AT SURGICAL MARGIN": "NO",
    "DYSPLASIA AT SURGICAL MARGIN (GRADING)": "MODERATE",
    "TUMOUR GRADING": "MODERATE",
    "PNI": "NO",
    "LVI": "NO",
    "PRIMARY TUMOUR SITE": "Tongue",
    "TYPE OF PRIMARY SURGICAL INTERVENTION": "Wide excision with neck dissection",
    "POST ADJUVANT THERAPY": "NO",
    "RECURRENCE (LR, LRR)": "NO",
}

CATEGORICAL_OPTIONS = {
    "SEX": ["MALE", "FEMALE"],
    "ETHNICITY": ["INDIAN", "MALAY", "CHINESE", "OTHER"],
    "ALCOHOL": ["YES", "NO"],
    "SMOKER": ["YES", "NO"],
    "BETEL NUT CHEWER": ["YES", "NO"],
    "COMORBID": ["YES", "NO"],
    "NODAL INVOLVEMENT": ["YES", "NO"],
    "ENE": ["YES", "NO"],
    "pTNM STAGING": ["I", "II", "III", "IV"],
    "VARIANT": ["CONVENTIONAL", "OTHER"],
    "LHR": ["STRONG", "WEAK", "ND"],
    "BONE INVASION": ["YES", "NO"],
    "BONE INVASION (TYPE)": ["ND", "TYPE 1", "TYPE 2"],
    "SURGICAL MARGIN": ["CLEAR", "CLOSE", "INVOLVED"],
    "DYSPLASIA AT SURGICAL MARGIN": ["YES", "NO"],
    "DYSPLASIA AT SURGICAL MARGIN (GRADING)": ["MILD", "MODERATE", "SEVERE", "ND"],
    "TUMOUR GRADING": ["LOW", "MODERATE", "HIGH"],
    "PNI": ["YES", "NO"],
    "LVI": ["YES", "NO"],
    "PRIMARY TUMOUR SITE": ["Tongue", "Buccal mucosa", "Floor of mouth", "Other"],
    "TYPE OF PRIMARY SURGICAL INTERVENTION": [
        "Wide excision with neck dissection",
        "Wide excision",
        "Hemiglossectomy",
        "Other",
    ],
    "POST ADJUVANT THERAPY": ["YES", "NO"],
    "RECURRENCE (LR, LRR)": ["YES", "NO"],
}

for key in FEATURE_COLUMNS:
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_VALUES[key]


def reset_defaults():
    for key in FEATURE_COLUMNS:
        st.session_state[key] = DEFAULT_VALUES[key]


st.title("Oral Squamous Cell Carcinoma Predictor")
st.write(
    "Select each feature value below, then review the prediction and SHAP explanation."
)

sidebar_col, main_col = st.columns([1, 3])
with sidebar_col:
    st.sidebar.button("Reset to default values", on_click=reset_defaults)

with main_col:
    tab1, tab2, tab3 = st.tabs(["Prediction", "Project Background", "How to Use"])

with tab1:
    st.subheader("Patient Inputs")

    # Keep the commented-out examples visible for reference.
    # 'Age': [user_age_input],
    # 'Sex': [user_sex_input],
    # 'Ki-67': [user_ki67_input],
    # ... all other inputs

    input_col1, input_col2 = st.columns(2)

    for i, feature in enumerate(FEATURE_COLUMNS):
        if feature in CATEGORICAL_OPTIONS:
            options = CATEGORICAL_OPTIONS[feature]
            current = st.session_state[feature]
            index = options.index(current) if current in options else 0
            st.session_state[feature] = input_col1.selectbox(
                feature,
                options=options,
                index=index,
                key=f"input_{feature}",
            )
        else:
            numeric_value = st.session_state[feature]
            if isinstance(numeric_value, str):
                numeric_value = 0
            st.session_state[feature] = input_col2.number_input(
                feature,
                value=float(numeric_value),
                step=1.0 if feature in ["AGE", "TUMOUR SIZE(pT)", "DOI(mm)", "TUMOUR DIMENSION(mm)"] else 0.1,
                key=f"input_{feature}",
            )

    user_data = pd.DataFrame([{col: st.session_state[col] for col in FEATURE_COLUMNS}])

    transformed_array = preprocessor.transform(user_data)
    feature_names = preprocessor.get_feature_names_out()
    shap_df = pd.DataFrame(transformed_array, columns=feature_names)

    prediction = model.predict(shap_df)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(shap_df)[0]
        prob = float(max(probabilities))
    else:
        probabilities = None
        prob = 0.0

    st.subheader("Prediction Result")
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted Class", str(prediction))
    with result_col2:
        st.metric("Confidence", f"{prob:.3f}")

    st.subheader("SHAP Explanation")
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(shap_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, shap_df, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP plot could not be displayed: {e}")

with tab2:
    st.subheader("Project Background")
    st.write(
        "This app is intended to support clinical review by combining patient data with an explainable machine learning model for oral squamous cell carcinoma outcomes."
    )
    st.write(
        "The interface provides a simple way to enter patient information, review predicted outcomes, and interpret the model with SHAP values."
    )

with tab3:
    st.subheader("How to Use")
    st.write(
        "1. Adjust values in the input panel.\n"
        "2. Use the reset button to restore default settings.\n"
        "3. Review the prediction and SHAP-based explanation in the results section."
    )
