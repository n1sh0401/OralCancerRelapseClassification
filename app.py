import time
import base64
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="OSCC Risk Predictor", layout="wide")

# Try to load a background image from the workspace (BG_Feathers.jpg). Fall back to a tiny transparent PNG.
base_path = Path(__file__).resolve().parent
img_file = base_path / "BG_Feathers.jpg"
if img_file.exists():
    try:
        with open(img_file, "rb") as f:
            bg_image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        bg_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQIW2NgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
else:
    bg_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQIW2NgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="

st.markdown(
    f"""
    <style>
    :root {{ --bg: #eceff3; --sidebar-bg: #f3f4f6; --muted: #9ca3af; --text-color: #000000; }}

    /* App background with subtle image */
    [data-testid="stAppViewContainer"] {{ 
        background-image: url('data:image/png;base64,{bg_image_base64}');
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-color: var(--bg);
        color: var(--text-color);
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(236,239,243,0.9);
        pointer-events: none;
        z-index: 0;
    }}
    [data-testid="stAppViewContainer"] * {{ color: var(--text-color) !important; position: relative; z-index: 1; }}

    /* Sidebar background and text */
    [data-testid="stSidebar"] > div {{ background-color: var(--sidebar-bg); color: var(--text-color); }}
    [data-testid="stSidebar"] > div * {{ color: var(--text-color) !important; }}

    /* Header / top bar: set dark background but white text for contrast */
    [data-testid="stHeader"] {{ background-color: #111827 !important; }}
    [data-testid="stHeader"] * {{ color: #ffffff !important; }}
    [data-testid="stHeader"] code {{ color: #ffffff !important; background-color: transparent !important; }}

    /* File uploader: ensure readable text */
    [data-testid="stFileUploader"] {{ background-color: var(--bg) !important; color: var(--text-color) !important; border: 1px dashed #e5e7eb; padding: 0.9rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(16,24,40,0.03); }}
    [data-testid="stFileUploader"] * {{ color: var(--text-color) !important; }}
    [data-testid="stFileUploader"] code {{ color: var(--text-color) !important; background-color: transparent !important; }}

    /* Text inputs, number inputs, selects, and textareas: dark background with white text for visibility */
    .stTextInput>div>input, .stNumberInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div>select, select {{
        background-color: #0f1724 !important;
        color: #ffffff !important;
        border: 1px solid #111827 !important;
        border-radius: 6px;
        padding: 6px 8px;
    }}
    .stTextArea>div>textarea::placeholder, .stTextInput>div>input::placeholder, .stNumberInput>div>input::placeholder {{
        color: rgba(255,255,255,0.7) !important;
    }}
    option {{ background-color: #0f1724; color: #ffffff; }}
    /* Ensure numeric inputs show white text too */
    input[type="number"], .stNumberInput>div>input, input[type="text"] {{
        color: #ffffff !important;
        background-color: #0f1724 !important;
    }}

    /* Dropdown menu option lists — Streamlit may render as divs/role=option/listbox */
    div[role="listbox"], div[role="listbox"] * , div[role="option"], div[role="option"] * {{
        color: #ffffff !important;
        background-color: #0f1724 !important;
    }}
    /* Additional selectors used by some Streamlit versions */
    .rc-virtual-list .rc-virtual-list-holder-inner div, .css-1v0mbdj, .css-1v0mbdj * {{
        color: #ffffff !important;
        background-color: #0f1724 !important;
    }}

    /* Inline code styling: light background with dark text */
    code {{ background-color: #f3f4f6; color: #000000; padding: 2px 6px; border-radius: 6px; }}

    .stButton>button {{ background-color: var(--bg); border: 1px solid #e5e7eb; color: var(--text-color); }}
    .stTextInput>div>input, .stTextArea>div>textarea {{ border: 1px solid #e9ecef; color: var(--text-color); background-color: var(--bg); }}
    .css-1kyxreq {{ padding: 0.6rem 0.8rem; }} /* small adjustments for layout */

    /* Keep links noticeable */
    a {{ color: #0b5fff !important; }}
    
    /* Force selected value text in Streamlit selectboxes/combo displays to be white */
    .stSelectbox>div, .stSelectbox>div * , div[role="combobox"], div[role="combobox"] * {{
        color: #ffffff !important;
    }}
    /* Cover a few extra possible containers Streamlit uses for the visible value */
    .stSelectbox .css-1v0mbdj, .stSelectbox .css-1v0mbdj * {{ color: #ffffff !important; }}
    .stSelectbox .css-1d0x0x8, .stSelectbox .css-1d0x0x8 * {{ color: #ffffff !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "oscc_LR_model.joblib"
    preprocessor_path = base_path / "preprocessor.joblib"
    model = joblib.load(model_path)

    # Compatibility shim: some sklearn versions changed internal helper class names
    # (e.g. _RemainderColsList) which can break unpickling. Define a lightweight
    # fallback in the sklearn.compose._column_transformer module so joblib can
    # reconstruct the preprocessor object. Prefer fixing sklearn versions in
    # production; this is a pragmatic runtime workaround.
    try:
        import sklearn.compose._column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Compatibility placeholder for older/newer sklearn pickles."""

                pass

            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        # If the shim cannot be applied, loading may still fail below.
        pass

    preprocessor = joblib.load(preprocessor_path)
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

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "prediction_error" not in st.session_state:
    st.session_state["prediction_error"] = None


def reset_defaults():
    for key in FEATURE_COLUMNS:
        st.session_state[key] = DEFAULT_VALUES[key]
    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"] = None


def render_input_fields(group_features):
    columns = st.columns(2)
    for idx, feature in enumerate(group_features):
        target_col = columns[idx % 2]
        if feature in CATEGORICAL_OPTIONS:
            options = CATEGORICAL_OPTIONS[feature]
            current = st.session_state[feature]
            index = options.index(current) if current in options else 0
            st.session_state[feature] = target_col.selectbox(
                feature,
                options=options,
                index=index,
                key=f"input_{feature}",
            )
        else:
            numeric_value = st.session_state[feature]
            if isinstance(numeric_value, str):
                numeric_value = 0
            st.session_state[feature] = target_col.number_input(
                feature,
                value=float(numeric_value),
                step=1.0 if feature in ["AGE", "TUMOUR SIZE(pT)", "DOI(mm)", "TUMOUR DIMENSION(mm)"] else 0.1,
                key=f"input_{feature}",
            )


def run_prediction():
    user_data = pd.DataFrame([{col: st.session_state[col] for col in FEATURE_COLUMNS}])

    for col in CATEGORICAL_OPTIONS.keys():
        if col in user_data.columns:
            user_data[col] = user_data[col].astype(str)

    numeric_cols = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_OPTIONS]
    original_values = user_data.copy()
    problematic_inputs = {}
    for col in numeric_cols:
        if col in user_data.columns:
            raw = user_data.loc[0, col]
            if (raw is None) or (str(raw).strip() == ""):
                user_data.loc[0, col] = DEFAULT_VALUES.get(col, 0)
            user_data[col] = pd.to_numeric(user_data[col], errors="coerce")
            if pd.isna(user_data.loc[0, col]):
                problematic_inputs[col] = original_values.loc[0, col]
                user_data.loc[0, col] = DEFAULT_VALUES.get(col, 0)

    nan_cols = user_data.columns[user_data.isnull().any()].tolist()
    if nan_cols:
        st.warning(f"Missing or invalid values in: {', '.join(nan_cols)}. Filling with defaults.")
        for col in nan_cols:
            if col in DEFAULT_VALUES:
                user_data[col].fillna(DEFAULT_VALUES[col], inplace=True)
            else:
                user_data[col].fillna(0, inplace=True)

    if problematic_inputs:
        msg_lines = [f"{k}: {v!s}" for k, v in problematic_inputs.items()]
        st.error("Inputs coerced to NaN before transform (original values shown); defaults were used:\n" + "; ".join(msg_lines))

    try:
        import numpy as _np
        from sklearn.pipeline import Pipeline as _Pipeline

        for _, transformer, cols in preprocessor.transformers_:
            if transformer in ("drop", "passthrough"):
                continue
            enc = transformer
            try:
                if isinstance(transformer, _Pipeline):
                    for _, step in reversed(transformer.steps):
                        if hasattr(step, "categories_"):
                            enc = step
                            break
            except Exception:
                pass
            if not hasattr(enc, "categories_"):
                continue

            if isinstance(cols, (list, tuple)):
                col_names = list(cols)
            else:
                try:
                    col_names = list(cols)
                except Exception:
                    col_names = [cols]

            for i, col in enumerate(col_names):
                if i >= len(enc.categories_):
                    break
                cats = enc.categories_[i]
                try:
                    cats_arr = _np.asarray(cats)
                    if cats_arr.dtype.kind in ("i", "u", "f", "b"):
                        raw = user_data.loc[0, col]
                        user_data[col] = pd.to_numeric(user_data[col], errors="coerce")
                        if pd.isna(user_data.loc[0, col]):
                            problematic_inputs[col] = original_values.loc[0, col]
                            user_data.loc[0, col] = DEFAULT_VALUES.get(col, 0)
                    else:
                        user_data[col] = user_data[col].astype(str)
                except Exception:
                    user_data[col] = user_data[col].astype(str)
    except Exception:
        pass

    try:
        import numpy as _np

        _np_isnan_orig = _np.isnan

        def _safe_isnan(x):
            try:
                return _np_isnan_orig(x)
            except TypeError:
                import pandas as _pd

                return _pd.isna(x)

        _np.isnan = _safe_isnan
        transformed_array = preprocessor.transform(user_data)
    finally:
        try:
            _np.isnan = _np_isnan_orig
        except Exception:
            pass

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    try:
        import numpy as _np

        if hasattr(transformed_array, "toarray"):
            transformed_dense = transformed_array.toarray()
        else:
            transformed_dense = _np.asarray(transformed_array)
    except Exception:
        transformed_dense = transformed_array

    has_nans = False
    try:
        import numpy as _np

        has_nans = _np.isnan(transformed_dense).any()
    except Exception:
        has_nans = False

    if has_nans:
        try:
            nan_positions = _np.argwhere(_np.isnan(transformed_dense)).tolist()
        except Exception:
            nan_positions = []
        st.error(f"Transformed data contains NaNs at positions: {nan_positions}. Check input coercion.")

    try:
        ncols = transformed_dense.shape[1]
    except Exception:
        ncols = 0

    if feature_names is not None and len(feature_names) == ncols:
        shap_df = pd.DataFrame(transformed_dense, columns=feature_names)
    else:
        shap_df = pd.DataFrame(transformed_dense, columns=[f"f_{i}" for i in range(ncols)])
        st.warning(f"Feature name count ({len(feature_names) if feature_names is not None else 'None'}) does not match transformed data columns ({ncols}). Using generic names.")

    prediction = model.predict(shap_df)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(shap_df)[0]
        prob = float(max(probabilities))
    else:
        probabilities = None
        prob = 0.0

    return prediction, prob, shap_df


st.title("Oral Squamous Cell Carcinoma Predictor")
st.write("Select each feature value below, then review the prediction and SHAP explanation.")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to", ["Prediction", "Project Background", "How to Use"], index=0)
    st.divider()

header_col, button_col = st.columns([4, 1])
with header_col:
    st.subheader("Patient Inputs")
with button_col:
    st.button("Reset Values", on_click=reset_defaults, use_container_width=True)

if section == "Prediction":
    st.write("Use the collapsible sections below to enter patient details. Click Predict when you are ready.")

    with st.expander("Demographics and lifestyle", expanded=False):
        render_input_fields(["SEX", "ETHNICITY", "ALCOHOL", "SMOKER", "BETEL NUT CHEWER"])

    with st.expander("Clinical and treatment details", expanded=False):
        render_input_fields([
            "AGE",
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
        ])

    predict_button = st.button("Predict", type="primary")
    if predict_button:
        st.info("Calculating...")
        time.sleep(0.5)
        try:
            prediction, prob, shap_df = run_prediction()
            st.session_state["prediction_result"] = (prediction, prob, shap_df)
            st.session_state["prediction_error"] = None
        except Exception as exc:
            st.session_state["prediction_result"] = None
            st.session_state["prediction_error"] = str(exc)

    if st.session_state.get("prediction_error"):
        st.error(f"Prediction could not be completed: {st.session_state['prediction_error']}")
    elif st.session_state.get("prediction_result") is not None:
        prediction, prob, shap_df = st.session_state["prediction_result"]
        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Predicted Class", str(prediction))
        with result_col2:
            st.metric("Confidence", f"{prob:.3f}")

        st.subheader("SHAP Explanation")
        try:
            # Prefer LinearExplainer for linear models (LogisticRegression).
            if hasattr(model, "coef_"):
                try:
                    explainer = shap.LinearExplainer(model, shap_df, feature_perturbation="interventional")
                    shap_vals = explainer.shap_values(shap_df)
                except Exception:
                    # Fallback to the general explainer using predict_proba
                    explainer = shap.Explainer(model.predict_proba, shap_df)
                    shap_vals = explainer(shap_df)

                # shap_vals from LinearExplainer may be an array or list (for multiclass)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    vals_to_plot = shap_vals[1]
                else:
                    vals_to_plot = shap_vals

                import numpy as _np
                # If shap returned only zeros (single-sample background issue), fall back to model coefficients
                vals_arr = _np.asarray(vals_to_plot)
                if _np.allclose(vals_arr, 0):
                    try:
                        coeffs = _np.asarray(model.coef_).ravel()
                        names = list(shap_df.columns)
                        if len(names) == len(coeffs):
                            idx = _np.argsort(_np.abs(coeffs))[::-1][:10]
                            top_names = [_np.array(names)[idx].tolist()]
                            top_coefs = coeffs[idx]
                            fig, ax = plt.subplots(figsize=(14, 6))
                            ax.barh(range(len(top_coefs))[::-1], top_coefs)
                            ax.set_yticks(range(len(top_coefs))[::-1])
                            ax.set_yticklabels([names[i] for i in idx])
                            ax.set_title('Top 10 model coefficients (fallback)')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning('SHAP values are zero and coefficient length does not match features; cannot show fallback.')
                    except Exception as exc:
                        st.warning(f'SHAP values are zero and coefficient fallback failed: {exc}')
                else:
                    with plt.rc_context({"ytick.labelsize": 6, "axes.titlesize": 8, "axes.labelsize": 6}):
                        fig, _ = plt.subplots(figsize=(14, 6))
                        shap.summary_plot(vals_to_plot, shap_df, plot_type="bar", show=False, max_display=10)
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                # For other models, explain the predict_proba output (works for classifiers)
                explainer = shap.Explainer(model.predict_proba, shap_df)
                shap_values = explainer(shap_df)

                # For binary classifiers, plot explanations for the positive class
                try:
                    class_idx = 1 if shap_values.values.shape[1] > 1 else 0
                    import numpy as _np
                    arr_vals = _np.asarray(getattr(shap_values, 'values', shap_values))
                    # select class axis if present
                    try:
                        class_vals = arr_vals[0, :, class_idx] if arr_vals.ndim == 3 else arr_vals
                    except Exception:
                        class_vals = arr_vals
                    if _np.allclose(class_vals, 0):
                        try:
                            coeffs = _np.asarray(model.coef_).ravel()
                            names = list(shap_df.columns)
                            if len(names) == len(coeffs):
                                idx = _np.argsort(_np.abs(coeffs))[::-1][:10]
                                top_coefs = coeffs[idx]
                                fig, ax = plt.subplots(figsize=(14, 6))
                                ax.barh(range(len(top_coefs))[::-1], top_coefs)
                                ax.set_yticks(range(len(top_coefs))[::-1])
                                ax.set_yticklabels([names[i] for i in idx])
                                ax.set_title('Top 10 model coefficients (fallback)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning('SHAP values are zero and coefficient length does not match features; cannot show fallback.')
                        except Exception as exc:
                            st.warning(f'SHAP values are zero and coefficient fallback failed: {exc}')
                    else:
                        with plt.rc_context({"ytick.labelsize": 6, "axes.titlesize": 8, "axes.labelsize": 6}):
                            fig, _ = plt.subplots(figsize=(14, 6))
                            shap.summary_plot(shap_values[:, class_idx], shap_df, plot_type="bar", show=False, max_display=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                except Exception:
                    import numpy as _np
                    arr_vals = _np.asarray(getattr(shap_values, 'values', shap_values))
                    if arr_vals.ndim == 3:
                        # sum across class axis as fallback
                        reduced = arr_vals[0].sum(axis=1)
                    else:
                        reduced = arr_vals.ravel()
                    if _np.allclose(reduced, 0):
                        try:
                            coeffs = _np.asarray(model.coef_).ravel()
                            names = list(shap_df.columns)
                            if len(names) == len(coeffs):
                                idx = _np.argsort(_np.abs(coeffs))[::-1][:10]
                                top_coefs = coeffs[idx]
                                fig, ax = plt.subplots(figsize=(14, 6))
                                ax.barh(range(len(top_coefs))[::-1], top_coefs)
                                ax.set_yticks(range(len(top_coefs))[::-1])
                                ax.set_yticklabels([names[i] for i in idx])
                                ax.set_title('Top 10 model coefficients (fallback)')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning('SHAP values are zero and coefficient length does not match features; cannot show fallback.')
                        except Exception as exc:
                            st.warning(f'SHAP values are zero and coefficient fallback failed: {exc}')
                    else:
                        with plt.rc_context({"ytick.labelsize": 6, "axes.titlesize": 8, "axes.labelsize": 6}):
                            fig, _ = plt.subplots(figsize=(14, 6))
                            shap.summary_plot(shap_values, shap_df, plot_type="bar", show=False, max_display=10)
                            plt.tight_layout()
                            st.pyplot(fig)
        except Exception as exc:
            st.warning(f"SHAP plot could not be displayed: {exc}")
    else:
        st.info("Click Predict to generate the outcome.")
elif section == "Project Background":
    st.subheader("Project Background")
    st.write(
        "This app is intended to support clinical review by combining patient data with an explainable machine learning model for oral squamous cell carcinoma outcomes."
    )
    st.write(
        "The interface provides a simple way to enter patient information, review predicted outcomes, and interpret the model with SHAP values."
    )
else:
    st.subheader("How to Use")
    st.write(
        "1. Adjust values in the input panel.\n"
        "2. Use the reset button to restore default settings.\n"
        "3. Review the prediction and SHAP-based explanation in the results section."
    )
