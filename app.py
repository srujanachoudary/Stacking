import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction using Stacking Regressor")

uploaded_file = st.file_uploader("Upload kc_house_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # One-hot encoding
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
        ('dt', DecisionTreeRegressor()),
        ('knn', KNeighborsRegressor()),
        ('rf', RandomForestRegressor()),
        ('svr', SVR())
    ]

    meta_model = LinearRegression()

    model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolu_
