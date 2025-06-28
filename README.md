# PRODIGY_ML_02
Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.
Features
📁 Data Overview: Inspect raw data, statistics, and missing values.

📊 Visualizations: See distributions, gender counts, and pairwise plots.

🔍 Clustering Analysis:

Use Elbow Method to find the optimal number of clusters.

Visualize clusters in 2D (Age vs Spending Score).

Explore interactive 3D Clusters (Age, Income, Spending Score).

How it works

Loads customer data (Mall_Customers.csv).

Uses @st.cache_data for efficient data loading.

Computes clusters with KMeans.

Plots cluster results in 2D and 3D.

Provides simple navigation with Streamlit’s sidebar.

