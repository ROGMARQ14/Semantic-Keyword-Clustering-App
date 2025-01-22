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
from io import StringIO

st.set_page_config(page_title="Keyword Clustering Tool", layout="wide")

st.title("Semantic Keyword Clustering App")

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
    value=0.85,
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

def create_chart(df, chart_type):
    """Create a sunburst chart or a treemap."""
    if chart_type == "sunburst":
        fig = px.sunburst(
            df, 
            path=['hub', 'spoke'], 
            values='cluster_size',
            color_discrete_sequence=px.colors.qualitative.Pastel2
        )
    else:  # treemap
        fig = px.treemap(
            df, 
            path=['hub', 'spoke'], 
            values='cluster_size',
            color_discrete_sequence=px.colors.qualitative.Pastel2
        )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig

def process_keywords(df, column_name):
    """Process the keywords and create clusters."""
    if column_name not in df.columns:
        st.error(f"The column name '{column_name}' is not in the uploaded file.")
        return None

    df.rename(columns={column_name: 'keyword'}, inplace=True)

    if REMOVE_DUPES:
        df.drop_duplicates(subset='keyword', inplace=True)

    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    from_list = df['keyword'].to_list()

    with st.spinner("Clustering keywords..."):
        embedding_model = SentenceTransformer(MODEL_NAME)
        distance_model = SentenceEmbeddings(embedding_model)

        model = PolyFuzz(distance_model)
        model = model.fit(from_list)
        model.group(link_min_similarity=MIN_SIMILARITY)

    df_cluster = model.get_matches()
    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    df = pd.merge(df, df_cluster[['keyword', 'spoke']], on='keyword', how='left')

    df['cluster_size'] = df['spoke'].map(df.groupby('spoke')['spoke'].count())
    df.loc[df["cluster_size"] == 1, "spoke"] = "no_cluster"
    df.insert(0, 'spoke', df.pop('spoke'))
    df['spoke'] = df['spoke'].str.encode('ascii', 'ignore').str.decode('ascii')

    df['keyword_len'] = df['keyword'].astype(str).apply(len)
    df = df.sort_values(by="keyword_len", ascending=True)

    df.insert(0, 'hub', df['spoke'].apply(create_unigram))

    df = df[['hub', 'spoke', 'cluster_size'] + 
            [col for col in df.columns if col not in ['hub', 'spoke', 'cluster_size']]]

    df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)
    df['spoke'] = (df['spoke'].str.split()).str.join(' ')

    return df

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Show available columns
        column_name = st.selectbox(
            "Select the keyword column",
            options=df.columns.tolist()
        )

        if st.button("Start Clustering"):
            start_time = time.time()
            
            # Process the keywords
            result_df = process_keywords(df, column_name)
            
            if result_df is not None:
                processing_time = time.time() - start_time
                st.success(f"Clustering completed in {processing_time:.2f} seconds!")

                # Create and display the visualization
                st.subheader("Cluster Visualization")
                fig = create_chart(result_df, CHART_TYPE)

                # Display statistics
                st.subheader("Clustering Statistics")
                total_keywords = len(result_df)
                total_clusters = len(result_df[result_df['spoke'] != 'no_cluster']['spoke'].unique())
                unclustered = len(result_df[result_df['spoke'] == 'no_cluster'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Keywords", total_keywords)
                col2.metric("Total Clusters", total_clusters)
                col3.metric("Unclustered Keywords", unclustered)

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
