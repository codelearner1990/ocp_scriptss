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

    dta['Internal/External'] = data['Internal/External'].str.lower().str.strip()

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
        return f"{int(minutes)} mins"
    elif minutes < 1440:  # Less than 24 hours
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours} hrs {mins} mins"
    else:  # 24 hours or more
        days = int(minutes // 1440)
        remaining_minutes = minutes % 1440
        hours = int(remaining_minutes // 60)
        mins = int(remaining_minutes % 60)
        return f"{days} days {hours} hrs {mins} mins"



def preprocess_traffic_impact(data, column_name='Traffic Impact'):
    # Fill NaN or blank values with empty strings
    data[column_name] = data[column_name].fillna("").astype(str)

    def parse_traffic_impact(traffic_str):
        impacts = {}
        for entry in traffic_str.split("\n"):
            if '=' in entry:
                category, items = entry.split('=', 1)
                category = category.strip()
                for item in items.split(','):
                    #try:
                        key, value = item.split(':', 1)
                        key = key.strip()
                        value = int(value.strip())
                        impacts[f"{category}_{key}"] = impacts.get(f"{category}_{key}", 0) + value
                    #except ValueError:
                        #st.warning(f"Skipping malformed entry: {item}")
            elif traffic_str.strip():
                #try:
                    generic_value = int(traffic_str.strip())
                    impacts[traffic_str.strip()] = impacts.get(traffic_str.strip(), 0) + generic_value
                #except ValueError:
                    st.warning(f"Skipping malformed generic entry: {traffic_str.strip()}")
        return impacts

    # Apply parsing logic to the column
    traffic_data = data[column_name].apply(parse_traffic_impact)

    # Flatten the parsed impacts and create a summary
    traffic_summary = Counter()
    for impact in traffic_data.dropna():
        traffic_summary.update(impact)

    # Convert summary into a DataFrame
    traffic_df = pd.DataFrame(traffic_summary.items(), columns=['Category', 'Count']).sort_values(by='Count', ascending=False)

    return traffic_df

# Function to preprocess and group traffic impact data
def preprocess_traffic_impact_with_granularity(data, column_name='Traffic Impact', granularity='Yearly'):
    # Preprocess Traffic Impact
    traffic_df = preprocess_traffic_impact(data, column_name)

    # Add granularity-specific columns
    if granularity == 'Yearly':
        data['Granularity'] = data['Year']
    elif granularity == 'Quarterly':
        data['Granularity'] = data['Quarter']
    elif granularity == 'Monthly':
        data['Granularity'] = data['Begin'].dt.to_period("M").astype(str)
    else:
        st.error("Unsupported granularity!")
        return None

    # Group traffic data by granularity and category
    grouped_wallet_data = (
        data.groupby(['Granularity', 'Category'])
        .size()
        .reset_index(name='Failure Count')
    )

    return grouped_wallet_data



# Streamlit application
st.title("Digital RRT Data Analysis")

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
    filtered_data['Root Cause Responsibility'] = filtered_data['Root Cause Responsibility'].apply(
        lambda x: x if ' ' nit in x else x.split('-')[0].rsplit('',1)[0]).str.strip()
    )
    filtered_data = filtered_data[filtered_data['Root Cause Responsibility']!= ""]
    if internal_external_filter != "Overall":
        filtered_data = filtered_data[filtered_data['Internal/External'].str.lower() == internal_external_filter.lower()]

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
        st.write("### 2. Root Cause Responsibility by Team")
        root_cause_responsibility = filtered_data['Root Cause Responsibility'].value_counts()
        threshold = 2
        if not root_cause_responsibility.empty:
            major_categories = root_cause_responsibility[root_cause_responsibility > threshold]
            fig, ax = plt.subplots()
            major_categories.plot(kind='bar', ax=ax, color='orange')
            ax.set_title('Root Cause Responsibility by Team')
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


        st.write(f"### 5. Impact by Wallet ({granularity})")


        if granularity == "Yearly":
            grouped_wallet_data = filtered_data.groupby(['Year', 'Category']).size().unstack(fill_value=0)
            x_col = 'Year'
        elif granularity == "Quarterly":
            grouped_wallet_data = filtered_data.groupby(['Quarter', 'Category']).size().unstack(fill_value=0)
            x_col = 'Quarter'
        elif granularity == "Monthly":
            filtered_data['Month-Year'] = filtered_data['Begin'].dt.to_period("M")
            grouped_wallet_data = filtered_data.groupby(['Month-Year', 'Category']).size().unstack(fill_value=0)
            x_col = 'Month-Year'

        # Plot the data if not empty
        if not grouped_wallet_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_wallet_data.plot(kind='bar', stacked=True, ax=ax, colormap="viridis", alpha=0.7)
            ax.set_title(f"Impact by Wallet ({granularity})")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Failure Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Display detailed data
            st.write(f"#### Detailed Data for Impact by Wallet ({granularity})")
            st.dataframe(grouped_wallet_data)
        else:
            st.warning("No data available for the selected filters.")


#Impact by wallet
# Improved Visualization of "Impact by Wallet"
st.write(f"### 5. Impact by Wallet ({granularity})")

# Generate grouped data for plotting
grouped_wallet_data = preprocess_traffic_impact_with_granularity(filtered_data, granularity=granularity)

if grouped_wallet_data is not None and not grouped_wallet_data.empty:
    # Sort data by failure count
    grouped_wallet_data = grouped_wallet_data.sort_values(by='Count', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=grouped_wallet_data,
        x='Count',
        y='Category',  # Horizontal bar chart
        hue='Granularity',  # Separate bars by granularity
        ax=ax,
        palette='tab10'
    )

    # Add data labels to bars
    for bar in ax.patches:
        ax.text(
            bar.get_width() + 50,  # Offset to avoid overlapping
            bar.get_y() + bar.get_height() / 2,
            f'{int(bar.get_width())}',
            ha='center', va='center', fontsize=10, color='black'
        )

    # Improve plot appearance
    ax.set_title(f"Impact by Wallet ({granularity})", fontsize=14, weight='bold')
    ax.set_xlabel("Failure Count", fontsize=12)
    ax.set_ylabel("Wallet", fontsize=12)
    ax.legend(title=granularity, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    st.pyplot(fig)

    # Display detailed data
    st.write(f"#### Detailed Data for Impact by Wallet ({granularity})")
    st.dataframe(grouped_wallet_data)
else:
    st.warning("No data available for the selected filters.")
