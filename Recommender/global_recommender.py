import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

def get_user_vector_from_csv(user_df, feature_cols):
    return user_df[feature_cols].mean(axis=0).values.reshape(1, -1)

def run_global_recommender():
    print("Music Recommendation System (Global Search) Initializing...")
    
    dataset_path = os.getenv("CLUSTERED_DATASET_PATH")
    user_input_path = os.getenv("USER_INPUT_PATH")
    
    if not dataset_path or not user_input_path:
        print("[ERROR] .env file could not be read or paths are missing!")
        return

    df = pd.read_csv(dataset_path)
    user_df = pd.read_csv(user_input_path)
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'speechiness', 'valence', 'loudness', 'tempo']
    
    user_vec = get_user_vector_from_csv(user_df, feature_cols)
    NUM_RECOMMENDATIONS = 20
    METRIC = 'cosine'
    
    X_all = df[feature_cols].values
    distances = pairwise_distances(user_vec, X_all, metric=METRIC)[0]
    nearest_indices = np.argsort(distances)[:NUM_RECOMMENDATIONS]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, 'global_recommendations.txt')
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("GLOBAL SEARCH RECOMMENDATION RESULTS\n")
        
        print(f"\nTop {NUM_RECOMMENDATIONS} songs found. Saving to '{output_filename}'...")
        
        for rank, idx in enumerate(nearest_indices, start=1):
            track_name = df.iloc[idx]['track_name']
            artist = df.iloc[idx]['artists']
            distance = distances[idx]
            
            print(f"{rank}. {track_name} - {artist}")
            f.write(f"{rank}. {track_name} - {artist}\n")
            
    print(f"\nSUCCESS! All recommendations have been saved to '{output_filename}'.")

if __name__ == "__main__":
    run_global_recommender()