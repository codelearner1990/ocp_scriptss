import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Cache data loading
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)
    data['Begin'] = pd.to_datetime(data['Begin'], errors='coerce')
    data['End'] = pd.to_datetime(data['End'], errors='coerce')
    data['Duration (mins)'] = (data['End'] - data['Begin']).dt.total_seconds() / 60
    data.fillna("Unknown", inplace=True)
    data['Year'] = data['Begin'].dt.year
    data['Month'] = data['Begin'].dt.month
    data['Quarter'] = data['Begin'].dt.to_period("Q")
    return data

# Cache pattern finding
@st.cache_data
def find_common_patterns(data, column_name, top_n=5):
    text_data = data[column_name].dropna()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    text_clusters = {}
    for i, text in enumerate(text_data):
        if i not in text_clusters:
            text_clusters[i] = []
        for j, score in enumerate(similarity_matrix[i]):
            if score > 0.8 and j != i:
                text_clusters[i].append(text_data.iloc[j])
    patterns = [', '.join(cluster) for cluster in text_clusters.values()]
    pattern_counts = Counter(patterns)
    return pattern_counts.most_common(top_n)

# Cache utility to format duration
def format_duration(minutes):
    if minutes < 60:
        return f"{minutes} mins"
    else:
        return f"{minutes // 60} hrs {minutes % 60} mins"

# Streamlit application
st.title("Comprehensive Incident Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your RRT Data (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    data = load_data(uploaded_file)

    # Global Filters
    st.sidebar.title("Global Filters")
    years = st.sidebar.multiselect("Select Years", data['Year'].unique(), default=data['Year'].unique())
    months = st.sidebar.multiselect("Select Months", data['Month'].unique(), default=data['Month'].unique())
    internal_external_filter = st.sidebar.radio("Select Scope", ["Overall", "Internal", "External"], index=0)
    granularity = st.sidebar.radio("Select Time Granularity", ["Yearly", "Quarterly", "Monthly"], index=1)

    # Apply filters
    filtered_data = data[(data['Year'].isin(years)) & (data['Month'].isin(months))]
    if internal_external_filter != "Overall":
        filtered_data = filtered_data[filtered_data['Internal/External'] == internal_external_filter]

    # Group data based on granularity
    if granularity == "Yearly":
        grouped_data = filtered_data.groupby('Year')[['MTTR', 'MTTD', 'MTTI']].sum().reset_index()
        x_col = 'Year'
    elif granularity == "Quarterly":
        grouped_data = filtered_data.groupby('Quarter')[['MTTR', 'MTTD', 'MTTI']].sum().reset_index()
        x_col = 'Quarter'
    elif granularity == "Monthly":
        filtered_data['Month-Year'] = filtered_data['Begin'].dt.to_period("M")
        grouped_data = filtered_data.groupby('Month-Year')[['MTTR', 'MTTD', 'MTTI']].sum().reset_index()
        x_col = 'Month-Year'

    # Raw Data Overview
    st.write("### Raw Data Overview")
    st.dataframe(filtered_data.head())

    # Question 1: How many issues caused by change?
    st.write("### 1. Issues Caused by Change")
    change_issues = filtered_data['Problem Caused By Change'].value_counts()
    fig, ax = plt.subplots()
    change_issues.plot(kind='bar', ax=ax, color='skyblue')
    for i, value in enumerate(change_issues):
        ax.text(i, value + 2, str(value), ha='center', fontsize=10)
    ax.set_title('Issues Caused by Change')
    ax.set_xlabel('Change Impact')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Drill-down for Question 1
    st.write("#### Drill Down to See Data")
    selected_category = st.selectbox("Select a Change Impact", options=change_issues.index)
    drill_down_data = filtered_data[filtered_data['Problem Caused By Change'] == selected_category]
    st.write(f"Filtered Data for '{selected_category}':")
    st.dataframe(drill_down_data)

    # Question 2: Root Cause Responsibility
    st.write("### 2. Root Cause Responsibility")
    root_cause_responsibility = filtered_data['Root Cause Responsibility'].value_counts()
    fig, ax = plt.subplots()
    root_cause_responsibility.plot(kind='bar', ax=ax, color='orange')
    ax.set_title('Root Cause Responsibility')
    ax.set_xlabel('Responsible Group')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Question 3: Avoidable Issues
    st.write("### 3. Avoidable Issues")
    avoidable_issues = filtered_data['Avoidability'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=avoidable_issues.index, y=avoidable_issues.values, ax=ax, palette='Blues_d')
    ax.set_title('Avoidable Issues')
    ax.set_xlabel('Avoidability')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Question 4: MTTR, MTTI, MTTD Distribution
    st.write(f"### MTTR, MTTI, MTTD Distribution ({granularity})")
    grouped_data['MTTR'] /= 60
    grouped_data['MTTI'] /= 60
    grouped_data['MTTD'] /= 60
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_data.plot(kind='bar', x=x_col, stacked=True, ax=ax, color=['red', 'green', 'blue'])
    ax.set_ylabel('Minutes')
    ax.set_title(f'MTTR, MTTI, MTTD Distribution ({granularity})')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Question 5: Common Patterns in Root Cause
    st.write("### 5. Common Patterns in Root Cause")
    root_cause_patterns = find_common_patterns(filtered_data, 'Root Cause', top_n=5)
    st.write("Common Patterns in Root Cause:", root_cause_patterns)

    # Display Detailed Data
    st.write("### Detailed Data")
    detailed_data = grouped_data.copy()
    detailed_data['MTTD'] = detailed_data['MTTD'].apply(lambda x: format_duration(int(x)))
    detailed_data['MTTI'] = detailed_data['MTTI'].apply(lambda x: format_duration(int(x)))
    detailed_data['MTTR'] = detailed_data['MTTR'].apply(lambda x: format_duration(int(x)))

    # Select only the required columns
    if granularity == "Monthly":
        detailed_data = detailed_data[['Month-Year', 'MTTD', 'MTTI', 'MTTR']]
    elif granularity == "Quarterly":
        detailed_data = detailed_data[['Quarter', 'MTTD', 'MTTI', 'MTTR']]
    else:  # Yearly
        detailed_data = detailed_data[['Year', 'MTTD', 'MTTI', 'MTTR']]

    # Display the cleaned-up detailed data
    st.dataframe(detailed_data)
