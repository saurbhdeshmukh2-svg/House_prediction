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
# CUSTOM CSS DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 45px;
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

/* Prediction Card */
.prediction-card {
    background: linear-gradient(
        135deg,
        #1e3a8a 0%,
        #2563eb 50%,
        #4f46e5 100%
    );

    padding: 35px;
    border-radius: 24px;
    text-align: center;
    color: white;

    box-shadow:
        0px 15px 35px rgba(37, 99, 235, 0.25);

    margin: 20px 0 30px 0;
}

.prediction-title {
    font-size: 20px;
    font-weight: 500;
    opacity: 0.9;
}

.prediction-price {
    font-size: 48px;
    font-weight: 800;
    margin: 8px 0;
}

.prediction-subtitle {
    font-size: 14px;
    opacity: 0.85;
}

/* Section Heading */
.section-title {
    font-size: 27px;
    font-weight: 750;
    color: #172554;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #eef2ff 100%
    );
}

/* Sidebar Heading */
[data-testid="stSidebar"] h1 {
    color: #172554;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    font-size: 14px;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #dbeafe;
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


model, scaler = load_model()


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

st.sidebar.markdown(
    "Enter the property information below."
)

st.sidebar.divider()


CRIM = st.sidebar.number_input(
    "CRIM - Crime Rate",
    min_value=0.0,
    value=0.10,
    format="%.4f"
)

ZN = st.sidebar.number_input(
    "ZN - Residential Land %",
    min_value=0.0,
    value=0.0
)

INDUS = st.sidebar.number_input(
    "INDUS - Business Land %",
    min_value=0.0,
    value=10.0
)

CHAS = st.sidebar.selectbox(
    "CHAS - Charles River",
    [0, 1]
)

NOX = st.sidebar.number_input(
    "NOX - Nitric Oxide",
    min_value=0.0,
    value=0.50,
    format="%.4f"
)

RM = st.sidebar.number_input(
    "RM - Average Rooms",
    min_value=0.0,
    value=6.00,
    format="%.2f"
)

AGE = st.sidebar.number_input(
    "AGE - House Age %",
    min_value=0.0,
    value=60.0
)

DIS = st.sidebar.number_input(
    "DIS - Employment Distance",
    min_value=0.0,
    value=4.00,
    format="%.4f"
)

RAD = st.sidebar.number_input(
    "RAD - Highway Accessibility",
    min_value=0.0,
    value=5.0
)

TAX = st.sidebar.number_input(
    "TAX - Property Tax",
    min_value=0.0,
    value=300.0
)

PTRATIO = st.sidebar.number_input(
    "PTRATIO - Pupil Teacher Ratio",
    min_value=0.0,
    value=18.0,
    format="%.2f"
)

B = st.sidebar.number_input(
    "B - Population Index",
    min_value=0.0,
    value=350.0,
    format="%.2f"
)

LSTAT = st.sidebar.number_input(
    "LSTAT - Lower Status %",
    min_value=0.0,
    value=12.0,
    format="%.2f"
)


# ============================================================
# FEATURE DATA
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
# MACHINE LEARNING PREDICTION
# ============================================================

try:

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    predicted_price = float(prediction[0])

except Exception as error:

    st.error(f"Prediction Error: {error}")
    st.stop()


# ============================================================
# PREDICTION RESULT CARD
# ============================================================

st.markdown(
    f"""
    <div class="prediction-card">

        <div class="prediction-title">
            Estimated House Price
        </div>

        <div class="prediction-price">
            ${predicted_price:,.2f}
        </div>

        <div class="prediction-subtitle">
            Machine Learning Prediction
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Property Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="🏠 Average Rooms",
        value=f"{RM:.2f}"
    )

with col2:

    st.metric(
        label="📍 Crime Rate",
        value=f"{CRIM:.4f}"
    )

with col3:

    st.metric(
        label="💰 Property Tax",
        value=f"{TAX:.0f}"
    )

with col4:

    st.metric(
        label="📊 Lower Status %",
        value=f"{LSTAT:.2f}%"
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
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


# ============================================================
# CHART 2 - IMPORTANT FEATURES
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
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
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
    '<div class="section-title">💰 Prediction Chart</div>',
    unsafe_allow_html=True
)

prediction_chart = go.Figure()

prediction_chart.add_trace(
    go.Bar(
        x=["Boston House"],
        y=[predicted_price],
        text=[f"${predicted_price:,.2f}"],
        textposition="outside",
        marker=dict(
            line=dict(
                width=1
            )
        )
    )
)

prediction_chart.update_layout(
    title="Predicted House Price",
    xaxis_title="Property",
    yaxis_title="Predicted Price",
    template="plotly_white",
    height=420,
    margin=dict(
        l=30,
        r=30,
        t=70,
        b=30
    )
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

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.info(
        f"🏠 **Predicted Price**\n\n"
        f"### ${predicted_price:,.2f}"
    )

with summary_col2:

    st.info(
        f"🛏️ **Average Rooms**\n\n"
        f"### {RM:.2f}"
    )

with summary_col3:

    st.info(
        f"📊 **Lower Status Population**\n\n"
        f"### {LSTAT:.2f}%"
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

        🏠 <b>Boston House Price Prediction</b>

        <br><br>

        Machine Learning Regression Project

        <br>

        Built with Python • Scikit-learn • Streamlit • Plotly

    </div>
    """,
    unsafe_allow_html=True
)