import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="House Price Stacking Model", layout="centered")

st.title("🏠 House Price Prediction (Stacking Model)")
st.write("Upload **kc_house_data.csv** to train and evaluate the model")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    categorical_features = ['zipcode']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    y = df['price']
    x = df.drop(['price', 'id', 'date'], axis=1)

    sc = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Base models
    base_models = [
        ('dt', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC())
    ]

    meta_model = LogisticRegression()

    classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

        st.success("Training completed!")

        st.subheader("📊 Model Performance")
        st.write(f"**Accuracy:** {accuracy:.4f}")

        st.subheader("Confusion Matrix")
        st.write(cm)
