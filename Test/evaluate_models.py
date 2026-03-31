import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from dotenv import load_dotenv, find_dotenv

# =====================================================================
# MAGIC CODE: Load global paths
# =====================================================================
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

# --- HELPER FUNCTIONS ---

def get_user_vector_from_csv(user_df, feature_cols):
    return user_df[feature_cols].mean(axis=0).values.reshape(1, -1)

def general_search(user_vector, X_all, n=20, metric='cosine'):
    distances = pairwise_distances(user_vector, X_all, metric=metric)
    nearest_indices = np.argsort(distances[0])[:n]
    return nearest_indices, distances[0][nearest_indices]

def optimized_cluster_search(user_vector, X_all, cluster_indices_dict, cluster_centers, n=20, metric='cosine'):
    center_distances = pairwise_distances(user_vector, cluster_centers, metric='euclidean')
    predicted_cluster = np.argmin(center_distances)
    
    subset_indices = cluster_indices_dict[predicted_cluster]
    X_subset = X_all[subset_indices]
    
    distances = pairwise_distances(user_vector, X_subset, metric=metric)
    
    local_nearest = np.argsort(distances[0])[:n]
    nearest_distances = distances[0][local_nearest]
    original_indices = subset_indices[local_nearest]
    
    return original_indices, nearest_distances

# --- MAIN BENCHMARKING SECTION ---

def run_benchmarks_and_plot():
    print("Initializing Benchmarks... Loading data and model.")
    
    dataset_path = os.getenv("CLUSTERED_DATASET_PATH")
    model_path = os.getenv("KMEANS_MODEL_PATH")
    user_input_path = os.getenv("USER_INPUT_PATH")
    
    if not dataset_path or not model_path or not user_input_path:
        print("\n[ERROR] .env file could not be read or paths are missing!")
        return

    df = pd.read_csv(dataset_path)
    model = joblib.load(model_path)
    user_df = pd.read_csv(user_input_path)
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'speechiness', 'valence', 'loudness', 'tempo']
    
    user_vec = get_user_vector_from_csv(user_df, feature_cols)
    N_REC = 20 
    
    # Offline Preparation
    X_numpy = df[feature_cols].values
    cluster_centers = model.cluster_centers_
    
    cluster_indices_dict = {}
    for c_id in range(model.n_clusters):
        cluster_indices_dict[c_id] = np.where(df['cluster_id'].values == c_id)[0]
        
    METRICS_TO_TEST = ['cosine', 'euclidean']
    
    # List to store results for our DataFrame
    benchmark_results = []
    
    for metric in METRICS_TO_TEST:
        print(f"\nTesting {metric.upper()} metric...")
        
        # --- GLOBAL SEARCH ---
        start_time = time.perf_counter()
        _, global_dists = general_search(user_vec, X_numpy, n=N_REC, metric=metric)
        end_time = time.perf_counter()
        
        global_time_ms = (end_time - start_time) * 1000
        global_avg_dist = np.mean(global_dists)
        
        benchmark_results.append({
            'Distance Metric': metric.capitalize(),
            'Search Method': 'Global (Brute Force)',
            'Search Time (ms)': global_time_ms,
            'Average Distance': global_avg_dist
        })
        
        # --- CLUSTER SEARCH ---
        start_time = time.perf_counter()
        _, cluster_dists = optimized_cluster_search(
            user_vec, X_numpy, cluster_indices_dict, cluster_centers, n=N_REC, metric=metric
        )
        end_time = time.perf_counter()
        
        cluster_time_ms = (end_time - start_time) * 1000
        cluster_avg_dist = np.mean(cluster_dists)
        
        benchmark_results.append({
            'Distance Metric': metric.capitalize(),
            'Search Method': 'Cluster (Optimized)',
            'Search Time (ms)': cluster_time_ms,
            'Average Distance': cluster_avg_dist
        })
        
        print(f"-> Global Time: {global_time_ms:.2f} ms | Cluster Time: {cluster_time_ms:.2f} ms")

    # =====================================================================
    # DATA VISUALIZATION (PLOTTING)
    # =====================================================================
    print("\nGenerating benchmark charts...")
    results_df = pd.DataFrame(benchmark_results)
    
    # Set styling
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Search Time Comparison (Speed)
    sns.barplot(
        data=results_df, 
        x='Distance Metric', 
        y='Search Time (ms)', 
        hue='Search Method', 
        ax=axes[0], 
        palette=['#e74c3c', '#2ecc71'] # Red for Global, Green for Cluster
    )
    axes[0].set_title('Search Time Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time in Milliseconds (ms)', fontsize=12)
    axes[0].set_xlabel('Distance Metric', fontsize=12)
    
    # Add data labels on top of the bars for Chart 1
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f', padding=3)

    # Chart 2: Average Distance Comparison (Quality)
    sns.barplot(
        data=results_df, 
        x='Distance Metric', 
        y='Average Distance', 
        hue='Search Method', 
        ax=axes[1], 
        palette=['#3498db', '#9b59b6'] # Blue and Purple
    )
    axes[1].set_title('Recommendation Quality / Distance (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Average Distance to User Profile', fontsize=12)
    axes[1].set_xlabel('Distance Metric', fontsize=12)
    
    # Add data labels on top of the bars for Chart 2
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.4f', padding=3)

    plt.tight_layout()
    
    # Save the figure to the same output directory where models are saved
    output_dir = os.path.dirname(os.path.abspath(__file__))
        
    plot_path = os.path.join(output_dir, 'results/performance_benchmark.png')
    plt.savefig(plot_path, dpi=300)
    
    print(f"SUCCESS! Benchmark visual saved to: {plot_path}")

if __name__ == "__main__":
    run_benchmarks_and_plot()