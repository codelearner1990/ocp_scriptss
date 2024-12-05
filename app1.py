import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

st.title("Optimized Incident Data Analysis")

# Cache the file loading process
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)
    data['Begin'] = pd.to_datetime(data['Begin'], errors='coerce')
    data['End'] = pd.to_datetime(data['End'], errors='coerce')
    data.fillna("Unknown", inplace=True)
    data['Duration (mins)'] = (data['End'] - data['Begin']).dt.total_seconds() / 60
    return data

# Cache pattern finding function
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

uploaded_file = st.file_uploader("Upload your RRT Data (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Raw Data Overview")
    st.dataframe(data.head(10))

    # Question 1: How many issues caused by change?
    st.write("### 1. Issues Caused by Change")
    change_issues = data['Problem Caused By Change'].value_counts()
    fig, ax = plt.subplots()
    change_issues.plot(kind='bar', ax=ax, color='skyblue')
    for i, value in enumerate(change_issues):
        ax.text(i, value + 2, str(value), ha='center', fontsize=10)
    ax.set_title('Issues Caused by Change')
    ax.set_xlabel('Change Impact')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Filter data dynamically for drill-down
    st.write("#### Drill Down to See Data")
    selected_category = st.selectbox("Select a Change Impact", options=change_issues.index)
    filtered_data = data[data['Problem Caused By Change'] == selected_category]
    st.write(f"Filtered Data for '{selected_category}':")
    st.dataframe(filtered_data)

    # Question 2: Graph based on Root Cause Responsibility
    st.write("### 2. Root Cause Responsibility")
    root_cause_responsibility = data['Root Cause Responsibility'].value_counts()
    fig, ax = plt.subplots()
    root_cause_responsibility.plot(kind='bar', ax=ax, color='orange')
    ax.set_title('Root Cause Responsibility')
    ax.set_xlabel('Responsible Group')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Question 3: What are all issues that can be avoided?
    st.write("### 3. Avoidable Issues")
    avoidable_issues = data['Avoidability'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=avoidable_issues.index, y=avoidable_issues.values, ax=ax, palette='Blues_d')
    ax.set_title('Avoidable Issues')
    ax.set_xlabel('Avoidability')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Question 4: Graph for MTTD, MTTI, MTTR and Correlation
# Allow users to filter by time period
    st.sidebar.title("Filter Options")
    period = st.sidebar.selectbox("Select Time Period", ["Quarterly", "Monthly", "Yearly", "Last Week"])

    # Group data based on selected period
    if period == "Quarterly":
        data['Quarter'] = data['Begin'].dt.to_period('Q').astype(str)
        grouped_data = data.groupby('Quarter')[['MTTR (mins)', 'MTTI (mins)', 'MTTD (mins)']].mean().reset_index()
        x_column = 'Quarter'
    elif period == "Monthly":
        data['Month-Year'] = data['Begin'].dt.to_period('M').astype(str)
        grouped_data = data.groupby('Month-Year')[['MTTR (mins)', 'MTTI (mins)', 'MTTD (mins)']].mean().reset_index()
        x_column = 'Month-Year'
    elif period == "Yearly":
        data['Year'] = data['Begin'].dt.year
        grouped_data = data.groupby('Year')[['MTTR (mins)', 'MTTI (mins)', 'MTTD (mins)']].mean().reset_index()
        x_column = 'Year'
    elif period == "Last Week":
        data['Week'] = data['Begin'].dt.to_period('W').astype(str)
        grouped_data = data.groupby('Week')[['MTTR (mins)', 'MTTI (mins)', 'MTTD (mins)']].mean().reset_index()
        grouped_data = grouped_data.tail(1)  # Only show the last week
        x_column = 'Week'

    # Stacked bar chart for MTTR, MTTI, and MTTD
    st.write(f"### MTTR, MTTI, and MTTD ({period})")
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_data.plot(
        x=x_column,
        kind='bar',
        stacked=True,
        color=['#FF9999', '#66B2FF', '#99FF99'],
        ax=ax
    )
    ax.set_title(f"MTTR, MTTI, MTTD Distribution ({period})")
    ax.set_xlabel(period)
    ax.set_ylabel("Time (mins)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Time Spent Breakdown (Pie Chart)
    st.write("### Total Time Breakdown (MTTR, MTTI, MTTD)")
    total_time = {
        "MTTR": data['MTTR (mins)'].sum(),
        "MTTI": data['MTTI (mins)'].sum(),
        "MTTD": data['MTTD (mins)'].sum(),
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        total_time.values(),
        labels=total_time.keys(),
        autopct="%1.1f%%",
        colors=['#FF9999', '#66B2FF', '#99FF99'],
        startangle=140
    )
    ax.set_title("Proportion of MTTR, MTTI, MTTD")
    st.pyplot(fig)

    # Display the grouped data table
    st.write("### Detailed Data")
    st.dataframe(grouped_data)

    # Question 5: Common Patterns in Root Cause
    st.write("### 5. Common Patterns in Root Cause")
    root_cause_patterns = find_common_patterns(data, 'Root Cause', top_n=5)
    st.write("Common Patterns in Root Cause:", root_cause_patterns)

    # Question 6: Number of Changes Caused by Certificates
    st.write("### 6. Issues Caused by Certificates")
    certificate_issues = data['Root Cause'].str.contains("certificate", case=False).sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [certificate_issues, len(data) - certificate_issues],
        labels=['Certificate Issues', 'Other Issues'],
        autopct='%1.1f%%',
        colors=sns.color_palette('pastel')
    )
    ax.set_title('Issues Caused by Certificates')
    st.pyplot(fig)

    # Question 7: Most Affected Applications
    st.write("### 7. Most Affected Applications")
    affected_applications = data['Business Service'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=affected_applications.index, y=affected_applications.values, ax=ax, palette="viridis")
    ax.set_title('Most Affected Applications')
    ax.set_xlabel('Application')
    ax.set_ylabel('Number of Issues')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Question 8: Different Types of Issues
    st.write("### 8. Different Types of Issues")
    issue_types = data['Type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=issue_types.index, y=issue_types.values, ax=ax, palette="coolwarm")
    ax.set_title('Different Types of Issues')
    ax.set_xlabel('Issue Type')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Remaining questions 9–13: Follow similar optimizations
    # Example: For restart-related issues, add caching and dynamic filtering
