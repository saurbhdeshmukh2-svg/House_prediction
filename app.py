import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef2ff 100%
    );
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 25px;
}

/* Section title */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #172554;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff,
        #eef2ff
    );
}

/* Metric styling */
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.07);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")

    return model, scaler


try:

    model, scaler = load_model()

except Exception as e:

    st.error("Unable to load model.joblib or scaler.joblib")
    st.error(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Boston House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict Boston house prices using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏡 Property Details")

st.sidebar.write(
    "Enter the property information:"
)

st.sidebar.divider()


# CRIM
CRIM = st.sidebar.number_input(
    "CRIM - Crime Rate",
    min_value=0.0,
    value=0.10,
    format="%.4f"
)


# ZN
ZN = st.sidebar.number_input(
    "ZN - Residential Land %",
    min_value=0.0,
    value=0.0
)


# INDUS
INDUS = st.sidebar.number_input(
    "INDUS - Business Land %",
    min_value=0.0,
    value=10.0
)


# CHAS
CHAS = st.sidebar.selectbox(
    "CHAS - Charles River",
    options=[0, 1]
)


# NOX
NOX = st.sidebar.number_input(
    "NOX - Nitric Oxide",
    min_value=0.0,
    value=0.50,
    format="%.4f"
)


# RM
RM = st.sidebar.number_input(
    "RM - Average Rooms",
    min_value=0.0,
    value=6.00,
    format="%.2f"
)


# AGE
AGE = st.sidebar.number_input(
    "AGE - House Age %",
    min_value=0.0,
    value=60.0
)


# DIS
DIS = st.sidebar.number_input(
    "DIS - Employment Distance",
    min_value=0.0,
    value=4.00,
    format="%.4f"
)


# RAD
RAD = st.sidebar.number_input(
    "RAD - Highway Accessibility",
    min_value=0.0,
    value=5.0
)


# TAX
TAX = st.sidebar.number_input(
    "TAX - Property Tax",
    min_value=0.0,
    value=300.0
)


# PTRATIO
PTRATIO = st.sidebar.number_input(
    "PTRATIO - Pupil Teacher Ratio",
    min_value=0.0,
    value=18.0,
    format="%.2f"
)


# B
B = st.sidebar.number_input(
    "B - Population Index",
    min_value=0.0,
    value=350.0,
    format="%.2f"
)


# LSTAT
LSTAT = st.sidebar.number_input(
    "LSTAT - Lower Status %",
    min_value=0.0,
    value=12.0,
    format="%.2f"
)


# ============================================================
# INPUT DATAFRAME
# ============================================================

features = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT"
]


values = [
    CRIM,
    ZN,
    INDUS,
    CHAS,
    NOX,
    RM,
    AGE,
    DIS,
    RAD,
    TAX,
    PTRATIO,
    B,
    LSTAT
]


input_data = pd.DataFrame(
    [values],
    columns=features
)


# ============================================================
# PREDICTION
# ============================================================

try:

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    predicted_price = float(prediction[0])

except Exception as e:

    st.error("Prediction failed.")
    st.error(str(e))
    st.stop()


# ============================================================
# PREDICTION RESULT
# ============================================================

st.markdown(
    '<div class="section-title">🏠 Prediction Result</div>',
    unsafe_allow_html=True
)


result_col1, result_col2, result_col3 = st.columns(3)


with result_col1:

    st.metric(
        label="🏠 Average Rooms",
        value=f"{RM:.2f}"
    )


with result_col2:

    st.metric(
        label="💰 Estimated House Price",
        value=f"${predicted_price:,.2f}"
    )


with result_col3:

    st.metric(
        label="📊 Lower Status %",
        value=f"{LSTAT:.2f}%"
    )


st.success("✅ Machine Learning Prediction")


# ============================================================
# PROPERTY OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Property Overview</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📍 Crime Rate",
        f"{CRIM:.4f}"
    )


with col2:

    st.metric(
        "🏠 Average Rooms",
        f"{RM:.2f}"
    )


with col3:

    st.metric(
        "💰 Property Tax",
        f"{TAX:.0f}"
    )


with col4:

    st.metric(
        "🛣️ Highway Access",
        f"{RAD:.0f}"
    )


st.divider()


# ============================================================
# FEATURE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">📈 Property Feature Analysis</div>',
    unsafe_allow_html=True
)


chart_col1, chart_col2 = st.columns(2)


# ============================================================
# CHART 1 - ALL FEATURES
# ============================================================

with chart_col1:

    feature_chart = pd.DataFrame({

        "Feature": [
            "CRIM",
            "ZN",
            "INDUS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT"
        ],

        "Value": [
            CRIM,
            ZN,
            INDUS,
            NOX,
            RM,
            AGE,
            DIS,
            RAD,
            TAX,
            PTRATIO,
            B,
            LSTAT
        ]
    })


    fig1 = go.Figure()


    fig1.add_trace(
        go.Bar(
            x=feature_chart["Feature"],
            y=feature_chart["Value"],
            text=feature_chart["Value"].round(2),
            textposition="outside"
        )
    )


    fig1.update_layout(
        title="Property Feature Values",
        xaxis_title="Features",
        yaxis_title="Value",
        template="plotly_white",
        height=450
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )


# ============================================================
# CHART 2 - KEY FEATURES
# ============================================================

with chart_col2:

    key_features = pd.DataFrame({

        "Feature": [
            "RM",
            "LSTAT",
            "PTRATIO",
            "NOX",
            "DIS"
        ],

        "Value": [
            RM,
            LSTAT,
            PTRATIO,
            NOX,
            DIS
        ]
    })


    fig2 = go.Figure()


    fig2.add_trace(
        go.Bar(
            x=key_features["Feature"],
            y=key_features["Value"],
            text=key_features["Value"].round(2),
            textposition="outside"
        )
    )


    fig2.update_layout(
        title="Key Property Indicators",
        xaxis_title="Features",
        yaxis_title="Value",
        template="plotly_white",
        height=450
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# PREDICTION CHART
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">💰 House Price Prediction Chart</div>',
    unsafe_allow_html=True
)


prediction_chart = go.Figure()


prediction_chart.add_trace(
    go.Bar(
        x=["Predicted House Price"],
        y=[predicted_price],
        text=[f"${predicted_price:,.2f}"],
        textposition="outside"
    )
)


prediction_chart.update_layout(
    title="Boston House Price Prediction",
    xaxis_title="Property",
    yaxis_title="Predicted Price",
    template="plotly_white",
    height=450
)


st.plotly_chart(
    prediction_chart,
    use_container_width=True
)


# ============================================================
# PRICE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">💵 Prediction Summary</div>',
    unsafe_allow_html=True
)


summary1, summary2, summary3 = st.columns(3)


with summary1:

    st.info(
        f"""
        ### 💰 Predicted Price

        **${predicted_price:,.2f}**
        """
    )


with summary2:

    st.info(
        f"""
        ### 🏠 Average Rooms

        **{RM:.2f}**
        """
    )


with summary3:

    st.info(
        f"""
        ### 📊 LSTAT

        **{LSTAT:.2f}%**
        """
    )


# ============================================================
# INPUT DATA TABLE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">📋 Input Data Summary</div>',
    unsafe_allow_html=True
)


display_data = input_data.T.reset_index()


display_data.columns = [
    "Feature",
    "Value"
]


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


with st.expander("⚙️ Model Information"):

    st.write("""
    **Project Name:** Boston House Price Prediction

    **Model Type:** Machine Learning Regression

    **Input Features:** 13

    **Output:** House Price

    **Model File:** model.joblib

    **Scaler File:** scaler.joblib

    **Framework:** Streamlit

    **Visualization:** Plotly
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🏠 Boston House Price Prediction

    <br>

    Machine Learning Regression Project

    <br><br>

    Built with Python • Scikit-learn • Streamlit • Plotly

    </div>
    """,
    unsafe_allow_html=True
)