import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (change the name to your actual file)
df = pd.read_csv('dataset.csv')

# Look at the raw data ranges
#print(df.describe())

# BEGIN DATA CLEANING********************************************************************************************************************
print(f"T0 (Original) shape: {df.shape}")
print(f"T0.1 (Original) info:\n{df.info()}")
## 1. Drop duplicates based on artist and track_name
df = df.drop_duplicates(subset=['artists', 'track_name'])
## 2. Drop rows with missing values
df = df.dropna()
print(f"T1 (duplications and NANs dropped) shape: {df.shape}")
print(f"T1.1 (duplications and NANs dropped) info:\n{df.info()}")

## CHECKING THE RANGES THAT SPECIFIED IN THE OFFICIAL SPOTIFY WEBSITE ------------------------------------------------
bounded_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
for col in bounded_cols:
    df = df[(df[col] >= 0) & (df[col] <= 1)]
print(f"T2 (built-in boundries checked) shape: {df.shape}")

## CHECKING THE RANGES ON RECOMMENDED HEURISTICS --------------------------------------------------------------------
### DURATION: Keep 1 min to 15 min
df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 900000)]
print(f"T3 (duration_ms filtered) shape: {df.shape}")

### TEMPO: Keep 40-230 BPM
df = df[(df['tempo'] >= 40) & (df['tempo'] <= 230)]
print(f"T4 (tempo filtered) shape: {df.shape}")

### LOUDNESS: Remove silence
df = df[df['loudness'] > -60]
print(f"T5 (loudness filtered) shape: {df.shape}")

### SPEECHINESS: Remove podcasts
df = df[df['speechiness'] < 0.66]
print(f"T6 (speechiness filtered) shape: {df.shape}")
# END DATA CLEANING********************************************************************************************************************

# BEGIN DATA NORMALIZATION---------------------------------------------------------------------------------------------------------------------------------------------------
print(f"T7 (statistics before normalization):\n{df.describe()}")
normalized_feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'loudness', 'tempo', 'popularity']
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[normalized_feature_cols] = scaler.fit_transform(df[normalized_feature_cols])
print(f"T8 (statistics after normalization)\n{df_normalized.describe()}")
# END DATA NORMALIZATION---------------------------------------------------------------------------------------------------------------------------------------------------

# BEGIN ONE HOT ENCODING FOR GENRE COLUMN
genre_encodes = pd.get_dummies(df_normalized['track_genre'], prefix='genre', dtype=int)
df_final = pd.concat([df_normalized, genre_encodes*0.2], axis=1)
print(f"T9 (after genre one-hot encoded) shape: {df_final.shape}")
# END ONE HOT ENCODING FOR GENRE COLUMN

df_final.to_csv('processed_dataset.csv', index=False)
print('Preprocessed Dataset saved.')