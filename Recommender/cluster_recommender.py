import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

def get_user_vector_from_csv(user_df, feature_cols):
    return user_df[feature_cols].mean(axis=0).values.reshape(1, -1)

def run_cluster_recommender():
    print("Music Recommendation System (Optimized Cluster Search) Initializing...")
    
    dataset_path = os.getenv("CLUSTERED_DATASET_PATH")
    model_path = os.getenv("KMEANS_MODEL_PATH")
    user_input_path = os.getenv("USER_INPUT_PATH")
    
    if not dataset_path or not model_path or not user_input_path:
        print("[ERROR] .env file could not be read or paths are missing!")
        return

    df = pd.read_csv(dataset_path)
    model = joblib.load(model_path)
    user_df = pd.read_csv(user_input_path)
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'speechiness', 'valence', 'loudness', 'tempo']
    
    user_vec = get_user_vector_from_csv(user_df, feature_cols)
    NUM_RECOMMENDATIONS = 20
    METRIC = 'cosine' 
    
    X_numpy = df[feature_cols].values
    cluster_centers = model.cluster_centers_
    
    cluster_indices_dict = {}
    for c_id in range(model.n_clusters):
        cluster_indices_dict[c_id] = np.where(df['cluster_id'].values == c_id)[0]
    
    center_distances = pairwise_distances(user_vec, cluster_centers, metric='euclidean')
    predicted_cluster = np.argmin(center_distances)
    
    subset_indices = cluster_indices_dict[predicted_cluster]
    X_subset = X_numpy[subset_indices]
    
    distances = pairwise_distances(user_vec, X_subset, metric=METRIC)[0]
    
    local_nearest = np.argsort(distances)[:NUM_RECOMMENDATIONS]
    nearest_distances = distances[local_nearest]
    original_indices = subset_indices[local_nearest]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, 'cluster_recommendations.txt')
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("CLUSTER SEARCH RECOMMENDATION RESULTS\n")
        
        print(f"\nTop {NUM_RECOMMENDATIONS} songs found in cluster {predicted_cluster}. Saving to '{output_filename}'...")
        
        for rank, (idx, dist) in enumerate(zip(original_indices, nearest_distances), start=1):
            track_name = df.iloc[idx]['track_name']
            artist = df.iloc[idx]['artists']
            
            print(f"{rank}. {track_name} - {artist}")
            f.write(f"{rank}. {track_name} - {artist}\n")
            
    print(f"\nSUCCESS! All recommendations have been saved to '{output_filename}'.")

if __name__ == "__main__":
    run_cluster_recommender()