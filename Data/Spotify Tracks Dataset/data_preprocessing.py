import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (change the name to your actual file)
df = pd.read_csv('dataset.csv')

# Look at the raw data ranges
print(df.describe())