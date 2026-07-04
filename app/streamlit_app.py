import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# =====================================================
# Page config
# =====================================================

st.set_page_config(
    page_title="Deposit Run Risk EWS",
    layout="wide"
)


# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_deposit_run_model.pkl")


# =====================================================
# Load data and model
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def create_features(df):
    df = df.copy()

    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["BalancePerProduct"] = df["Balance"] / (df["NumOfProducts"] + 1)
    df["AgeTenureRatio"] = df["Age"] / (df["Tenure"] + 1)

    median_balance = df["Balance"].median()

    df["HighBalanceInactive"] = (
        (df["Balance"] > median_balance)
        & (df["IsActiveMember"] == 0)
    ).astype(int)

    df["CustomerValue"] = df["Balance"] * df["NumOfProducts"]

    return df


def assign_risk_band(prob):
    if prob < 0.20:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Critical"


# =====================================================
# Prepare predictions
# =====================================================

df_raw = load_data()
model = load_model()

df = create_features(df_raw)

drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
X = df.drop(columns=drop_cols)

df["Runoff_Probability"] = model.predict_proba(X)[:, 1]
df["Risk_Band"] = df["Runoff_Probability"].apply(assign_risk_band)
df["Deposit_At_Risk"] = df["Balance"] * df["Runoff_Probability"]

risk_order = ["Low", "Medium", "High", "Critical"]

df["Risk_Band"] = pd.Categorical(
    df["Risk_Band"],
    categories=risk_order,
    ordered=True
)


# =====================================================
# Header
# =====================================================

st.title("Banking Deposit Run Risk Early Warning System")

st.markdown("""
This dashboard identifies customers with elevated deposit runoff risk and estimates
expected deposits at risk to support liquidity risk monitoring and proactive customer management.
""")


# =====================================================
# Sidebar filters
# =====================================================

st.sidebar.header("Filters")

risk_band_filter = st.sidebar.multiselect(
    "Risk Band",
    options=risk_order,
    default=risk_order
)

geography_filter = st.sidebar.multiselect(
    "Geography",
    options=sorted(df["Geography"].unique()),
    default=sorted(df["Geography"].unique())
)

active_filter = st.sidebar.multiselect(
    "Active Member",
    options=[0, 1],
    default=[0, 1]
)

filtered = df[
    (df["Risk_Band"].astype(str).isin(risk_band_filter))
    & (df["Geography"].isin(geography_filter))
    & (df["IsActiveMember"].isin(active_filter))
].copy()

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


# =====================================================
# KPI cards
# =====================================================

total_customers = len(filtered)
total_balance = filtered["Balance"].sum()
total_deposit_at_risk = filtered["Deposit_At_Risk"].sum()
avg_runoff_prob = filtered["Runoff_Probability"].mean()
critical_customers = (filtered["Risk_Band"].astype(str) == "Critical").sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Customers", f"{total_customers:,.0f}")
col2.metric("Total Balance", f"£{total_balance:,.0f}")
col3.metric("Deposit at Risk", f"£{total_deposit_at_risk:,.0f}")
col4.metric("Avg Runoff Probability", f"{avg_runoff_prob:.1%}")
col5.metric("Critical Customers", f"{critical_customers:,.0f}")


if total_deposit_at_risk > 40_000_000:
    st.error("Deposit at Risk is above the monitoring threshold.")
elif total_deposit_at_risk > 25_000_000:
    st.warning("Deposit at Risk is material and should be actively monitored.")
else:
    st.success("Deposit at Risk is within the acceptable monitoring range.")


# =====================================================
# Portfolio summary
# =====================================================

st.header("Portfolio Risk Summary")

summary = (
    filtered
    .groupby("Risk_Band", observed=False)
    .agg(
        Customer_Count=("CustomerId", "count"),
        Total_Balance=("Balance", "sum"),
        Deposit_At_Risk=("Deposit_At_Risk", "sum"),
        Avg_Runoff_Prob=("Runoff_Probability", "mean")
    )
    .reset_index()
)

summary["Pct_of_Total_Deposit_At_Risk"] = (
    summary["Deposit_At_Risk"] / summary["Deposit_At_Risk"].sum()
)

st.dataframe(
    summary.style.format({
        "Total_Balance": "£{:,.0f}",
        "Deposit_At_Risk": "£{:,.0f}",
        "Avg_Runoff_Prob": "{:.1%}",
        "Pct_of_Total_Deposit_At_Risk": "{:.1%}"
    }),
    use_container_width=True
)


# =====================================================
# Charts
# =====================================================

left, right = st.columns(2)

with left:
    fig_customer = px.bar(
        summary,
        x="Risk_Band",
        y="Customer_Count",
        title="Customer Count by Risk Band",
        text_auto=True,
        category_orders={"Risk_Band": risk_order}
    )
    st.plotly_chart(fig_customer, use_container_width=True)

with right:
    fig_deposit = px.bar(
        summary,
        x="Risk_Band",
        y="Deposit_At_Risk",
        title="Deposit at Risk by Risk Band",
        text_auto=".2s",
        category_orders={"Risk_Band": risk_order}
    )
    st.plotly_chart(fig_deposit, use_container_width=True)


fig_pie = px.pie(
    summary,
    names="Risk_Band",
    values="Deposit_At_Risk",
    title="Contribution to Total Deposit at Risk",
    category_orders={"Risk_Band": risk_order}
)

st.plotly_chart(fig_pie, use_container_width=True)


# =====================================================
# Top customers
# =====================================================

st.header("Top Customers by Deposit at Risk")

top_n = st.slider(
    "Number of customers to display",
    min_value=5,
    max_value=50,
    value=20
)

top_customers = (
    filtered
    .sort_values("Deposit_At_Risk", ascending=False)
    .head(top_n)
)

st.dataframe(
    top_customers[
        [
            "CustomerId",
            "Geography",
            "Gender",
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "NumOfProducts",
            "IsActiveMember",
            "Runoff_Probability",
            "Risk_Band",
            "Deposit_At_Risk"
        ]
    ].style.format({
        "Balance": "£{:,.0f}",
        "EstimatedSalary": "£{:,.0f}",
        "Runoff_Probability": "{:.1%}",
        "Deposit_At_Risk": "£{:,.0f}"
    }),
    use_container_width=True
)


# =====================================================
# Customer drill-down
# =====================================================

st.header("Customer Drill-down")

customer_id = st.selectbox(
    "Select Customer",
    options=top_customers["CustomerId"].tolist()
)

customer = filtered[filtered["CustomerId"] == customer_id].iloc[0]

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("Runoff Probability", f"{customer['Runoff_Probability']:.1%}")
col_b.metric("Balance", f"£{customer['Balance']:,.0f}")
col_c.metric("Deposit at Risk", f"£{customer['Deposit_At_Risk']:,.0f}")
col_d.metric("Risk Band", str(customer["Risk_Band"]))

if customer["Runoff_Probability"] >= 0.80:
    st.error("This customer is classified as Critical Risk.")
elif customer["Runoff_Probability"] >= 0.50:
    st.warning("This customer is classified as High Risk.")
elif customer["Runoff_Probability"] >= 0.20:
    st.info("This customer is classified as Medium Risk.")
else:
    st.success("This customer is classified as Low Risk.")


st.write("Customer Profile")

customer_profile = pd.DataFrame(customer).T[
    [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "CreditScore",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "Risk_Band"
    ]
]

st.dataframe(
    customer_profile,
    use_container_width=True
)


# =====================================================
# Executive insight
# =====================================================

st.header("Executive Insight")

st.markdown("""
- Runoff probability alone does not fully capture liquidity impact.
- Deposit at Risk combines customer-level probability with balance exposure.
- Medium-risk customers may contribute materially to expected runoff due to their larger population size.
- This dashboard can support Treasury, Liquidity Risk and Relationship Management teams in prioritising customer monitoring and retention actions.
""")


