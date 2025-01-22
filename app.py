import streamlit as st
import time
import pandas as pd
import os
from collections import Counter
from sentence_transformers import SentenceTransformer
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="Keyword Clustering Tool App", layout="wide")

st.title("Semantic Keyword Clustering")

# Sidebar controls
st.sidebar.header("Settings")

# Model selection
MODEL_OPTIONS = {
    "Fast (Lower Accuracy)": "paraphrase-MiniLM-L3-v2",
    "Balanced": "all-MiniLM-L6-v2",
    "Accurate (Slower)": "all-mpnet-base-v2"
}
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)
MODEL_NAME = MODEL_OPTIONS[selected_model]

# Similarity threshold
MIN_SIMILARITY = st.sidebar.slider(
    "Minimum Similarity Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.90,
    step=0.05
)

# Chart type selection
CHART_TYPE = st.sidebar.selectbox(
    "Visualization Type",
    options=["treemap", "sunburst"],
    index=0
)

# Remove duplicates option
REMOVE_DUPES = st.sidebar.checkbox("Remove Duplicates", value=True)

def create_unigram(cluster: str):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    most_common_word = Counter(words).most_common(1)[0][0]
    return most_common_word

@st.cache_resource
def get_model(model_name: str):
    """Create and return a SentenceTransformer model."""
    with st.spinner("Loading the model..."):
        model = SentenceTransformer(model_name)
    return model

def create_chart(df, chart_type, has_volume=False):
    """Create a sunburst chart or a treemap."""
    # Determine values column based on volume availability
    values_column = 'cluster_volume' if has_volume and 'cluster_volume' in df.columns else 'cluster_size'
    
    if chart_type == "sunburst":
        fig = px.sunburst(
            df, 
            path=['hub', 'spoke'], 
            values=values_column,
            color_discrete_sequence=px.colors.qualitative.Pastel2
        )
    else:  # treemap
        fig = px.treemap(
            df, 
            path=['hub', 'spoke'], 
            values=values_column,
            color_discrete_sequence=px.colors.qualitative.Pastel2
        )
    
    # Update title based on values column
    title = "Clusters by Search Volume" if values_column == 'cluster_volume' else "Clusters by Keyword Count"
    fig.update_layout(title_text=title)
    
    st.plotly_chart(fig, use_container_width=True)
    return fig

def process_keywords(df, column_name, volume_column=None):
    """Process the keywords and create clusters."""
    if column_name not in df.columns:
        st.error(f"The column name '{column_name}' is not in the uploaded file.")
        return None

    # Create a copy of the dataframe with required columns
    working_df = df[[column_name]].copy()
    if volume_column and volume_column in df.columns:
        working_df[volume_column] = df[volume_column]
        # Convert volume to numeric, replacing any non-numeric values with 0
        working_df[volume_column] = pd.to_numeric(working_df[volume_column], errors='coerce').fillna(0)
    
    working_df.rename(columns={column_name: 'keyword'}, inplace=True)

    if REMOVE_DUPES:
        if volume_column:
            # If removing duplicates with volume, keep the one with highest volume
            working_df = working_df.sort_values(volume_column, ascending=False).drop_duplicates(subset='keyword')
        else:
            working_df.drop_duplicates(subset='keyword', inplace=True)

    working_df = working_df[working_df["keyword"].notna()]
    working_df['keyword'] = working_df['keyword'].astype(str)
    from_list = working_df['keyword'].to_list()

    with st.spinner("Clustering keywords..."):
        embedding_model = SentenceTransformer(MODEL_NAME)
        distance_model = SentenceEmbeddings(embedding_model)

        model = PolyFuzz(distance_model)
        model = model.fit(from_list)
        model.group(link_min_similarity=MIN_SIMILARITY)

    df_cluster = model.get_matches()
    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    working_df = pd.merge(working_df, df_cluster[['keyword', 'spoke', 'similarity']], on='keyword', how='left')

    # Calculate cluster metrics
    if volume_column:
        working_df['cluster_size'] = working_df.groupby('spoke')['spoke'].transform('count')
        working_df['cluster_volume'] = working_df.groupby('spoke')[volume_column].transform('sum')
    else:
        working_df['cluster_size'] = working_df['spoke'].map(working_df.groupby('spoke')['spoke'].count())
    
    working_df.loc[working_df["cluster_size"] == 1, "spoke"] = "no_cluster"
    working_df.insert(0, 'spoke', working_df.pop('spoke'))
    working_df['spoke'] = working_df['spoke'].str.encode('ascii', 'ignore').str.decode('ascii')

    working_df['keyword_len'] = working_df['keyword'].astype(str).apply(len)
    working_df = working_df.sort_values(by="keyword_len", ascending=True)

    working_df.insert(0, 'hub', working_df['spoke'].apply(create_unigram))

    # Reorder columns based on whether volume exists
    base_columns = ['hub', 'spoke', 'cluster_size']
    if volume_column:
        base_columns.extend(['cluster_volume', volume_column])
    
    remaining_columns = [col for col in working_df.columns if col not in base_columns]
    working_df = working_df[base_columns + remaining_columns]

    working_df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)
    working_df['spoke'] = (working_df['spoke'].str.split()).str.join(' ')

    return working_df

def read_file(uploaded_file):
    """Read either CSV or Excel file with proper handling of delimiters."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        # Try different delimiters
        try:
            # First try to read with pandas auto delimiter detection
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except:
            try:
                # Try with specific delimiters
                for delimiter in [',', ';', '|', '\t']:
                    try:
                        df = pd.read_csv(uploaded_file, sep=delimiter)
                        if len(df.columns) > 1:  # Successfully found correct delimiter
                            break
                    except:
                        continue
            except:
                st.error("Could not read the CSV file. Please check the file format.")
                return None
    elif file_type in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            st.error("Could not read the Excel file. Please check the file format.")
            return None
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    return df

# Main app layout
st.write("Upload your CSV or Excel file containing keywords and optional search volume data.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = read_file(uploaded_file)
        if df is not None:
            st.success("File uploaded successfully!")
            
            # Simple column selection
            st.subheader("Select Columns")
            
            # Display preview of the data
            st.write("Preview of your data:")
            st.dataframe(df.head())
            
            # Keyword column selection
            keyword_col = st.selectbox(
                "Select the column containing your keywords",
                options=df.columns.tolist()
            )
            
            # Volume column selection (optional)
            volume_options = ['None'] + [col for col in df.columns if col != keyword_col]
            volume_col = st.selectbox(
                "Select the search volume column (optional)",
                options=volume_options
            )
            
            has_volume = volume_col != 'None'
            
            if st.button("Start Clustering"):
                start_time = time.time()
                
                # Process the keywords
                vol_col = volume_col if has_volume else None
                result_df = process_keywords(df, keyword_col, vol_col)
                
                if result_df is not None:
                    processing_time = time.time() - start_time
                    st.success(f"Clustering completed in {processing_time:.2f} seconds!")

                    # Create and display the visualization
                    st.subheader("Cluster Visualization")
                    fig = create_chart(result_df, CHART_TYPE, has_volume)

                    # Display statistics
                    st.subheader("Clustering Statistics")
                    total_keywords = len(result_df)
                    total_clusters = len(result_df[result_df['spoke'] != 'no_cluster']['spoke'].unique())
                    unclustered = len(result_df[result_df['spoke'] == 'no_cluster'])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Keywords", total_keywords)
                    col2.metric("Total Clusters", total_clusters)
                    col3.metric("Unclustered Keywords", unclustered)

                    if has_volume:
                        st.subheader("Volume Statistics")
                        total_volume = result_df[volume_col].sum()
                        avg_volume = total_volume / total_keywords if total_keywords > 0 else 0
                        max_cluster_volume = result_df[result_df['spoke'] != 'no_cluster'].groupby('spoke')[volume_col].sum().max()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Search Volume", f"{total_volume:,.0f}")
                        col2.metric("Average Volume per Keyword", f"{avg_volume:,.0f}")
                        col3.metric("Largest Cluster Volume", f"{max_cluster_volume:,.0f}")

                    # Download buttons
                    st.subheader("Download Results")
                    
                    # CSV download
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="clustered_keywords.csv",
                        mime="text/csv"
                    )
                    
                    # HTML chart download
                    html_bytes = pio.to_html(fig, include_plotlyjs=True)
                    st.download_button(
                        label="Download Interactive Chart",
                        data=html_bytes,
                        file_name=f"keyword_clusters_{CHART_TYPE}.html",
                        mime="text/html"
                    )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Add installation instructions for Colab
with st.expander("📝 Installation Instructions for Google Colab"):
    st.code("""
# Run these commands in a Colab cell before running the app
!pip install streamlit polyfuzz sentence-transformers plotly
!npm install localtunnel

# In a new cell, save the code above as streamlit_app.py and run:
!streamlit run streamlit_app.py & npx localtunnel --port 8501
    """, language="bash")
