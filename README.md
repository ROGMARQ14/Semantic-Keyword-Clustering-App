# Semantic Keyword Clustering App

A Streamlit application that performs semantic clustering on keywords using transformer-based models. This tool helps in organizing and visualizing large sets of keywords based on their semantic similarity.

## Features

- 🔍 Semantic keyword clustering using state-of-the-art transformer models
- 📊 Interactive visualizations (Treemap and Sunburst charts)
- 🎯 Adjustable similarity threshold
- 🚀 Multiple model options (balancing speed vs accuracy)
- 📈 Clustering statistics and metrics
- 💾 Export results as CSV and interactive HTML charts

## Demo

You can access the live demo at: [Your Streamlit App URL]

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd keyword-clustering
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Upload your CSV file containing keywords
3. Select the column containing your keywords
4. Configure the clustering settings:
   - Choose a model (Fast, Balanced, or Accurate)
   - Adjust the similarity threshold
   - Select visualization type
   - Toggle duplicate removal
5. Click "Start Clustering" to begin the process
6. Download results as CSV or interactive HTML visualization

## Input Format

The app accepts CSV files with the following requirements:
- Must contain at least one column with keywords
- File should be UTF-8 encoded
- No specific row limit, but larger datasets will take longer to process

## Models Available

1. **Fast (Lower Accuracy)**: paraphrase-MiniLM-L3-v2
   - Best for quick analysis and large datasets
   - Lower memory usage

2. **Balanced**: all-MiniLM-L6-v2
   - Good balance between speed and accuracy
   - Recommended for most use cases

3. **Accurate (Slower)**: all-mpnet-base-v2
   - Best semantic matching
   - Requires more processing time and memory

## Deployment

To deploy on Streamlit Cloud:

1. Fork this repository
2. Log in to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to your forked repository
4. Select the main file as `streamlit_app.py`

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and feature requests, please create an issue in the repository.
