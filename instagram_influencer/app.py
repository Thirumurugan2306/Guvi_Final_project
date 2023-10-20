import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

# Load the dataset (assuming it's in a CSV file named 'Influencer.csv')
df = pd.read_csv('instagram_influencer/Influencer.csv')

# Rename columns and clean data
column_mapping = {col: col.replace(' ', '_') for col in df.columns}
df.rename(columns=column_mapping, inplace=True)
df.columns = df.columns.str.lower()
df.rename(columns={'avg._likes': 'avg_likes', 'new_post_avg._likes': 'new_post_avg_likes', '60-day_eng_rate': '60_day_eng_rate'}, inplace=True)
df['channel_info'] = df['channel_info'].str.strip().str.replace(r'^\\', '', regex=True)

# Calculate the mode of the 'country_or_region' column
mode_country = df['country_or_region'].mode()[0]
df['country_or_region'].fillna(mode_country, inplace=True)

# Define a function to convert value strings to numeric values
def convert_value(value):
    if isinstance(value, str):
        value = value.lower()
        if 'm' in value:
            return float(value.replace('m', '')) * 1e6
        elif 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif 'b' in value:
            return float(value.replace('b', '')) * 1e9
        elif re.match(r'^\d+\.\d+[kK]$', value):
            return float(value.replace('k', '')) * 1e3
    try:
        return float(value)
    except ValueError:
        return value

# Convert columns with suffixes to float
columns_to_convert = ['followers', 'avg_likes', 'new_post_avg_likes', 'total_likes']
for column in columns_to_convert:
    df[column] = df[column].apply(convert_value)

# Streamlit app
st.title('Instagram Influencer Analysis')


# Sidebar for selecting visuals
selected_visual = st.sidebar.selectbox("Select Visual", ["Frequency Distribution - Influence Score",
                                                        "Frequency Distribution - Followers",
                                                        "Frequency Distribution - Posts",
                                                        "Number of Influencers by Country",
                                                        "Top 10 Influencers by Followers",
                                                        "Correlation Heatmap",
                                                        "Scatter Plots"])

# Display selected visuals based on the selection
if selected_visual == "Frequency Distribution - Influence Score":
    st.subheader('Frequency Distribution of Influence Score')
    fig = px.histogram(df, x='influence_score', nbins=20, title='Frequency Distribution of Influence Score')
    st.plotly_chart(fig)

elif selected_visual == "Frequency Distribution - Followers":
    st.subheader('Frequency Distribution of Followers')
    fig = px.histogram(df, x='followers', nbins=20, title='Frequency Distribution of Followers')
    st.plotly_chart(fig)

elif selected_visual == "Frequency Distribution - Posts":
    st.subheader('Frequency Distribution of Posts')
    fig = px.histogram(df, x='posts', nbins=20, title='Frequency Distribution of Posts')
    st.plotly_chart(fig)

elif selected_visual == "Number of Influencers by Country":
    st.subheader('Number of Instagram Influencers by Country')
    country_counts = df['country_or_region'].value_counts()
    fig = px.bar(country_counts, y=country_counts.index, x=country_counts.values, orientation='h',
                 title='Number of Instagram Influencers by Country')
    st.plotly_chart(fig)

elif selected_visual == "Top 10 Influencers by Followers":
    st.subheader('Top 10 Influencers based on Followers')
    top_followers = df.sort_values(by='followers', ascending=False).head(10)
    fig = px.bar(top_followers, x='followers', y='channel_info', orientation='h',
                 title='Top 10 Influencers based on Followers')
    st.plotly_chart(fig)

elif selected_visual == "Correlation Heatmap":
    st.subheader('Correlation - Heatmap')
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_columns.corr()
    st.plotly_chart(px.imshow(correlation_matrix, color_continuous_scale='blues'))

elif selected_visual == "Scatter Plots":
    st.subheader('Scatter Plots')
    scatter_fig1 = px.scatter(df, x='followers', y='total_likes', title='Relationship between Followers and Total Likes')
    scatter_fig2 = px.scatter(df, x='followers', y='influence_score', title='Relationship between Followers and Influence Score')
    scatter_fig3 = px.scatter(df, x='posts', y='avg_likes', title='Relationship between Posts and Average Likes')
    scatter_fig4 = px.scatter(df, x='posts', y='influence_score', title='Relationship between Posts and Influence Score')

    st.plotly_chart(scatter_fig1)
    st.plotly_chart(scatter_fig2)
    st.plotly_chart(scatter_fig3)
    st.plotly_chart(scatter_fig4)
