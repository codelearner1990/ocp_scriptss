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
    # Load data
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    # Ensure 'Begin' and 'End' columns are properly converted to datetime
    for col in ['Begin', 'End']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')  # Invalid dates will be set as NaT (Not a Time)

    # Add 'Duration (mins)' column
    if 'Begin' in data.columns and 'End' in data.columns:
        data['Duration (mins)'] = (data['End'] - data['Begin']).dt.total_seconds() / 60

    # Fill missing values based on column data type
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            data[column] = data[column].fillna(0).astype(float)  # Ensure numeric columns are float
        elif data[column].dtype == 'object' or pd.api.types.is_string_dtype(data[column]):
            data[column] = data[column].fillna("Unknown").astype(str)  # Ensure string columns remain string

    # Add 'Year', 'Month', and 'Quarter' columns if 'Begin' is valid
    if 'Begin' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Begin']):
        data['Year'] = data['Begin'].dt.year
        data['Month'] = data['Begin'].dt.month
        data['Quarter'] = data['Begin'].dt.to_period("Q")
    else:
        # Fallback in case 'Begin' is missing or invalid
        data['Year'] = "Unknown"
        data['Month'] = "Unknown"
        data['Quarter'] = "Unknown"

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


def parse_traffic_impact(traffic_str):
    impacts = {}
    for entry in traffic_str.split("\n"):
        if '=' in entry:
            category, items = entry.split("=", 1)
            if ':' in items:
                for item in items.split(","):
                    try:
                        key, value = item.split(":")
                        key = key.strip()
                        value = value.strip()
                        if key and value.isdigit():  # Check if key is not empty and value is a valid integer
                            impacts[f"{category.strip()}_{key}"] = impacts.get(f"{category.strip()}_{key}", 0) + int(value)
                    except ValueError:
                        st.warning(f"Skipping malformed entry: {item}")
            else:
                try:
                    generic_value = items.strip()
                    if generic_value.isdigit():  # Check if the generic value is a valid integer
                        impacts[category.strip()] = impacts.get(category.strip(), 0) + int(generic_value)
                except ValueError:
                    st.warning(f"Skipping malformed generic entry: {items}")
    return impacts


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
    filtered_data['Root Cause Responsibility'] = filtered_data['Root Cause Responsibility'].str.split('-').str[0].str.strip()
    if internal_external_filter != "Overall":
        filtered_data = filtered_data[filtered_data['Internal/External'] == internal_external_filter]

    if filtered_data.empty:
        st.error("No data available for the selected filters. Please adjust your filters.")
    else:
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
        if not change_issues.empty:
            fig, ax = plt.subplots()
            change_issues.plot(kind='bar', ax=ax, color='skyblue')
            for i, value in enumerate(change_issues):
                ax.text(i, value + 2, str(value), ha='center', fontsize=10)
            ax.set_title('Issues Caused by Change')
            ax.set_xlabel('Change Impact')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.warning("No data for 'Issues Caused by Change'.")

        # Question 2: Root Cause Responsibility
        st.write("### 2. Root Cause Responsibility")
        root_cause_responsibility = filtered_data['Root Cause Responsibility'].value_counts()
        threshold = 2
        if not root_cause_responsibility.empty:
            major_categories = root_cause_responsibility[root_cause_responsibility > threshold]
            fig, ax = plt.subplots()
            major_categories.plot(kind='bar', ax=ax, color='orange')
            ax.set_title('Root Cause Responsibility')
            ax.set_xlabel('Responsible Group')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No data for 'Root Cause Responsibility'.")

        # Question 3: Avoidable Issues
        st.write("### 3. Avoidable Issues")
        avoidable_issues = filtered_data['Avoidability'].value_counts()
        if not avoidable_issues.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=avoidable_issues.index, y=avoidable_issues.values, ax=ax, palette='Blues_d')
            ax.set_title('Avoidable Issues')
            ax.set_xlabel('Avoidability')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.warning("No data for 'Avoidable Issues'.")

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

        # MTTR Over Time (Updated Line Chart)
 # MTTR Over Time (Updated Line Chart)
st.write(f"### MTTR, MTTI, MTTD Over Time ({granularity})")

# Convert MTTR, MTTD, and MTTI into days, hours, and minutes dynamically
def convert_time_column(data, column):
    data[f'{column}_formatted'] = data[column].apply(lambda x: f"{x // (24 * 60)}d {(x % (24 * 60)) // 60}h {x % 60}m" if x >= 1440 else
                                                     f"{x // 60}h {x % 60}m" if x >= 60 else f"{x}m")
    return data

monthly_data = filtered_data.groupby(['Year', 'Month'])[['MTTR', 'MTTI', 'MTTD']].sum().reset_index()
monthly_data['Month-Year'] = monthly_data['Year'].astype(str) + "-" + monthly_data['Month'].astype(str)

# Apply dynamic formatting
monthly_data = convert_time_column(monthly_data, 'MTTR')
monthly_data = convert_time_column(monthly_data, 'MTTI')
monthly_data = convert_time_column(monthly_data, 'MTTD')

# Line plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=monthly_data, x='Month-Year', y='MTTR', marker='o', label='MTTR', ax=ax, color='red')
sns.lineplot(data=monthly_data, x='Month-Year', y='MTTI', marker='o', label='MTTI', ax=ax, color='green')
sns.lineplot(data=monthly_data, x='Month-Year', y='MTTD', marker='o', label='MTTD', ax=ax, color='blue')

# Formatting the x-axis
ax.set_xticklabels(monthly_data['Month-Year'], rotation=45)
ax.set_title(f"MTTR, MTTI, MTTD Over Time ({granularity})")
ax.set_ylabel("Minutes")
ax.legend(loc="upper left")

# Display plot
st.pyplot(fig)

# Display Detailed Data (Optional)
st.write("### Detailed MTTR, MTTI, MTTD Over Time")
st.dataframe(monthly_data[['Month-Year', 'MTTR_formatted', 'MTTI_formatted', 'MTTD_formatted']])



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

# Question: Number of Issues by Priority
st.write(f"### 9. Number of Issues by Priority ({granularity})")

# Group data based on selected granularity
if granularity == "Yearly":
    grouped_priority_data = filtered_data.groupby(['Year', 'Priority']).size().reset_index(name='Count')
    x_col = 'Year'
elif granularity == "Quarterly":
    grouped_priority_data = filtered_data.groupby(['Quarter', 'Priority']).size().reset_index(name='Count')
    x_col = 'Quarter'
elif granularity == "Monthly":
    filtered_data['Month-Year'] = filtered_data['Begin'].dt.to_period("M")
    grouped_priority_data = filtered_data.groupby(['Month-Year', 'Priority']).size().reset_index(name='Count')
    x_col = 'Month-Year'

# Check if there is data to plot
if not grouped_priority_data.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=grouped_priority_data,
        x=x_col,
        y='Count',
        hue='Priority',  # Add Priority as a hue to separate bars
        ax=ax,
        palette='muted'
    )
    ax.set_title(f"Number of Issues by Priority ({granularity})")
    ax.set_xlabel(granularity)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("No data available for the selected filters.")
