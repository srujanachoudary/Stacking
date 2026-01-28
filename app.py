import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction (Stacking Regressor)")

uploaded_file = st.file_uploader("Upload kc_house_data.csv", type=["csv"])

@st.cache_resource
def train_model(df):
    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

    y = df['price']
    x = df.drop(['price', 'id', 'date'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    base_models = [
        ('dt', DecisionTreeRegressor(max_depth=8, random_state=42)),
        ('rf', RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]

    model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(),
        cv=3
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded")

    if st.button("🚀 Train Model"):
        with st.spinner("Training model (cloud-safe)..."):
            r2, mae = train_model(df)

        st.subheader("📊 Results")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MAE:** ₹ {mae:,.2f}")
