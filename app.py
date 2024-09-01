import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
def load_data():
    df = pd.read_csv(r'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
    X = df.iloc[:,:-1]  # All columns except the last one
    y = df.iloc[:,-1]   # Last column
    return X, y

# Data preparation
X, y = load_data()
X = X.values
y = y.values

lb = LabelEncoder()
y_encoded = lb.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model
rfc = RandomForestClassifier()
model = rfc.fit(X_train, y_train)

# Streamlit app
st.title("Iris Species Classifier")

# Create sliders for input
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# When the user clicks the button, the prediction is made
if st.button("Classify"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_class = lb.inverse_transform(prediction)
    
    st.write(f"The predicted species is: **{predicted_class[0]}**")





