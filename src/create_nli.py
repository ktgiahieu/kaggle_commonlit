import pandas as pd
from sklearn import model_selection

df1 = pd.read_csv("/content/kaggle_commonlit/data/train.csv")
df2 = pd.read_csv("/content/kaggle_commonlit/data/train.csv")
#df = df.dropna().reset_index(drop=True)
# df = df.merge(df, how='outer',on='id')
df1['key'] = 0
df2['key'] = 0

df = df1.merge(df2, on='key', how='outer')
df = df[df['id_x']!=df['id_y']]
df['target'] = (df['target_x'] > df['target_y']).apply(lambda x:int(x))
df = df[['id_x', 'excerpt_x', 'id_y', 'excerpt_y','target']]

df = df.sample(n=10000, random_state=50898).reset_index(drop=True)


# df.to_csv("/content/kaggle_commonlit/data/train_nli_folds.csv", index=False)

train_df, valid_df = model_selection.train_test_split(df, test_size=0.2)
train_df.to_csv("/content/kaggle_commonlit/data/train_nli_folds.csv", index=False)
valid_df.to_csv("/content/kaggle_commonlit/data/valid_nli_folds.csv", index=False)
    
