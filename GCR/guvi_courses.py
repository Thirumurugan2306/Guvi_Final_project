import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.grid import grid

df=pd.read_csv('GCR/guvi_courses.csv')

unique_courses=df['course_title'].unique()
unique_subject=df['subject'].unique()
unique_courses=df['course_title'].unique()

# Load the best regression model
best_model = joblib.load('best_regression_model.pkl')

st.set_page_config(
        page_title="Airbnb Data Analysis",
        layout="wide",
    )

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1633613286991-611fe299c4be?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
        background-attachment: scroll;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Define CSS styles for the title
title_style = """
    color: white;
    text-align: center;
    padding: 10px;
    background-color: Green;
    border-radius: 15px; /* Adjust the value to control the curvature */
"""

# Display the Guvi logo with a link
st.markdown(
    f'<a href="https://www.guvi.in/">'
    f'<img src="https://www.guvi.in/build/images/guvi-logo.e8ad68fbd8dc0a5fc2f7c4ffd580c54d.png" alt="Guvi Logo" width="200px">'
    f'</a>',
    unsafe_allow_html=True
)
st.markdown(f'<h1 style="{title_style}">Guvi Courses - Rating Predictor</h1>', unsafe_allow_html=True)


# Input form for user to enter course details


course_title = st.selectbox("Course Title",unique_courses )
subject = st.selectbox("Subject",unique_subject)
my_grid = grid(2,2, vertical_align="bottom")
level = my_grid.select_slider("Level", ["Beginner", "Intermediate", "Advanced"])
price = my_grid.slider("Price",0,200,1)
num_subscribers = my_grid.number_input("Number of Subscribers",value=100, placeholder="Type a no of subribers...")
num_reviews = my_grid.number_input("Number of Reviews",value=100, placeholder="Type a no of reviews...")
content_duration=my_grid.number_input("select the content duration",value=10, placeholder="Enter the duration in minutes...")
num_lectures = my_grid.number_input("Number of Lectures",value=10, placeholder="Enter the no of lectures...")


# Encode categorical variables
encoder = LabelEncoder()
level_encoded = encoder.fit_transform([level])
subject_encoded = encoder.fit_transform([subject])

# Convert 'published_timestamp' to days since publication
current_date = datetime.now()
days_since_published = (current_date - current_date).days  # Replace current_date with actual published date

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'price': [price],
    'num_subscribers': [num_subscribers],
    'num_reviews': [num_reviews],
    'num_lectures': [num_lectures],
    'level': level_encoded,
    'content_duration':content_duration,
    'subject': subject_encoded,
    'days_since_published': [days_since_published]
})

# Ensure the feature order and names match
feature_order = ['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'level','content_duration', 'subject', 'days_since_published']
input_data = input_data[feature_order]

col1, col2, col3 = st.columns(3)


# Predict the course rating using the model
if col2.button("Predict Rating"):
    # Make the prediction
    predicted_rating = best_model.predict(input_data)
    
    # Map the predicted rating to a star rating
    star_rating = int((predicted_rating[0] * 5) + 1)  # Assuming the predicted rating is between 0 and 1
    
    # Display the predicted course rating and star rating to the user
    st.success(f"Predicted Course Rating: {predicted_rating[0]:.2f}")
    st.success(f"Star Rating: {star_rating} stars")
    st.link_button("Go to Guvi site for more courses", "https://www.guvi.in/")

