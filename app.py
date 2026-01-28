import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Social Network Ads Prediction", layout="centered")

st.title("📊 Social Network Ads – Purchase Prediction")
st.write("Stacking Classifier using Age and Estimated Salary")

# Upload CSV
uploaded_file = st.file_uploader("Upload Social_Network_Ads.csv", type=["csv"])

@st.cache_resource
def train_model(df):
    # Features and target
    x = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Base models
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]

    # Meta model
    meta_model = LogisticRegression(max_iter=1000)

    # Stacking classifier
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("🚀 Train Model"):
        with st.spinner("Training stacking classifier..."):
            accuracy, cm = train_model(df)

        st.subheader("📈 Model Performance")
        st.write(f"**Accuracy:** {accuracy:.4f}")

        st.subheader("Confusion Matrix")
        st.write(cm)
