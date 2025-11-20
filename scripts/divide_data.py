import pandas as pd


df = pd.read_parquet('../data/processed/CLASF_data.parquet')

vars_to_drop = ['title', 'text', 'user_id', 'timestamp', 'helpful_vote','verified_purchase']
df = df.drop(columns=vars_to_drop)

df_short = df.sample(n=120_000)

idx = set(df.index)

short_df_idx = set(df_short.index)
long_df_idx = idx - short_df_idx
df_long = df.iloc[list(long_df_idx),:]

print(len(long_df_idx))
print(len(idx) - 120_000)

#df_long = df[]
df_short.to_parquet('../data/processed/CLASF_short_data.parquet')
df_long.to_parquet('../data/processed/CLASF_long_data.parquet')
