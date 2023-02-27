import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import train_test_split



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('sales.jpg')  

# Load the sales data
sales_df = pd.read_csv('sales_data.csv')

# Create the feature matrix and target vector
X = sales_df[['item_name', 'day']]
y = sales_df['sales']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['day', 'item_name'], prefix=['day', 'item_name'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the linear regression model
model = LinearRegression()

# Train the model on the sales data
model.fit(X_train, y_train)

# Calculate the mean squared error and R^2 score for the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error: ', mse)
print('R^2 score:', r2)

# Get input from user
item_name =st.selectbox(
     'select item name',
    ('vada', 'samoosa', 'cream bun','pazhampori','bajji'))
day = st.selectbox(
    'select day',
    ('Monday', 'Tuesday', 'Wednesday','Thursday','Friday'))

# Create new input
X_new = pd.DataFrame({'item_name': [item_name], 'day': [day]})
X_new = pd.get_dummies(X_new, columns=['day', 'item_name'], prefix=['day', 'item_name'])

# Add missing dummy variables
missing_cols = set(X.columns) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0

# Ensure columns are in the same order
X_new = X_new[X.columns]

# Predict the sales for the new input
y_new = model.predict(X_new)
if st.button('Predict'):
    st.write('Predicted sales: ', y_new[0])
    
  


