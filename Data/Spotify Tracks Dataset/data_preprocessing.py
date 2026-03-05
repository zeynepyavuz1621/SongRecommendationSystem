import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (change the name to your actual file)
df = pd.read_csv('dataset.csv')

# Look at the raw data ranges
#print(df.describe())

print(f"Original shape: {df.shape}")
# 1. Drop duplicates based on artist and track_name
df = df.drop_duplicates(subset=['artists', 'track_name'])
# 2. Drop rows with missing values
df = df.dropna()
print(f"T1 shape: {df.shape}")
df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 600000)]
print(f"T2 shape: {df.shape}")

df['combined_features'] = (
df['artists'].fillna('') + ' ' +
df['album_name'].fillna('') + ' ' +
df['track_name'].fillna('')
)

print(df['combined_features'].head)