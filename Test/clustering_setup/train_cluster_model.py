import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from dotenv import load_dotenv, find_dotenv

# =====================================================================
# MAGIC CODE: Finds the .env file and loads global paths
# =====================================================================
load_dotenv(find_dotenv())

# Fetch paths from the .env file
input_data_path = os.getenv("PREPROCESSED_DATA_PATH")
output_dataset_path = os.getenv("CLUSTERED_DATASET_PATH")
output_model_path = os.getenv("KMEANS_MODEL_PATH")

# Security Check: Was .env read successfully?
if not input_data_path or not output_dataset_path or not output_model_path:
    print("\n[ERROR] .env file could not be read or paths are missing! Please check your .env file in the root directory.")
    exit()

# Dynamically find the output directory for models, datasets, and plots
# (e.g., extracts "clustering_setup" from "clustering_setup/clustered_dataset.csv")
cluster_output_dir = os.path.dirname(output_dataset_path)

# Automatically create the directory if it doesn't exist (prevents errors)
if cluster_output_dir:
    os.makedirs(cluster_output_dir, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(input_data_path)

# Select only continuous audio features
feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'speechiness', 'valence', 'loudness', 'tempo']
X = df[feature_cols]

# =====================================================================
# STAGE 1: FINDING THE NUMBER OF CLUSTERS (K) (ELBOW METHOD)
# =====================================================================
print("\n--- Calculating Elbow Method (for K values 1-15) ---")
k_values = range(1, 16)
wcss_scores = []

for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto') 
    kmeans_temp.fit(X)
    wcss_scores.append(kmeans_temp.inertia_)
    print(f"Trial: K={k:2d} | Inertia = {kmeans_temp.inertia_:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss_scores, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)

# Save the elbow plot to the relevant output directory
elbow_plot_path = os.path.join(cluster_output_dir, 'elbow_curve_analysis.png')
plt.savefig(elbow_plot_path)
print(f"-> Elbow plot saved: {elbow_plot_path}")

# =====================================================================
# STAGE 2: TRAINING AND SAVING THE FINAL MODEL
# =====================================================================
optimal_k = 6  

print(f"\n--- Training final model with optimal_k = {optimal_k} ---")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
df['cluster_id'] = final_kmeans.fit_predict(X)

# =====================================================================
# STAGE 3: REDUCING CLUSTERS TO 2D USING PCA AND VISUALIZING
# =====================================================================
print("\n--- Reducing clusters to 2D with PCA and visualizing ---")
# Reduce 9-dimensional X data to 2 dimensions (X, Y coordinates)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Temporarily add these 2 dimensions to the dataframe for easy plotting
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]

# Draw colored scatter plot (by cluster_id) using Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df, 
    x='pca_x', 
    y='pca_y', 
    hue='cluster_id', 
    palette='tab10', # Color palette
    s=15,            # Point size
    alpha=0.6        # Transparency
)
plt.title(f'K-Means Clustering (K={optimal_k}) - 2D PCA View')
plt.xlabel('First Principal Component (PCA 1)')
plt.ylabel('Second Principal Component (PCA 2)')
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the PCA plot to the output directory
pca_plot_path = os.path.join(cluster_output_dir, 'cluster_visualization_pca.png')
plt.savefig(pca_plot_path, dpi=300)
print(f"-> Clustering plot saved in high resolution: {pca_plot_path}")

# Drop temporary PCA columns before saving to disk
df = df.drop(columns=['pca_x', 'pca_y'])

# Save the clustered dataset and model to the paths specified in .env
df.to_csv(output_dataset_path, index=False)
joblib.dump(final_kmeans, output_model_path)

print(f"\nSUCCESS! Model and data generated in the '{cluster_output_dir}' directory.")