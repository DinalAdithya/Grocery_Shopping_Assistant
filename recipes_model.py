import pandas as pd
import ast # used convert string list to actual list
import pickle # to save & load Ml models so i dont have to retrain them
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("PP_recipes.csv")# load data set

df = df[['id', 'name_tokens', 'ingredient_tokens']]
# to keep usefull cloumns and drop others
df.dropna(inplace=True)

# to convert ingredient_tokens list into actual lists
df['ingredient_tokens'] = df['ingredient_tokens'].apply(ast.literal_eval)

# to convert ingredient_tokens list to text format
df['ingredient_tokens'] = df['ingredient_tokens'].apply(lambda x: ' '.join(map(str, x)))

#convert ingredient_tokens into numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['ingredient_tokens'])

# Train recipe recommendation model
model = NearestNeighbors(n_neighbors=3, metric="cosine")
model.fit(X)

#
with open("recipe_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("model training comlete! Saved as 'recipe_model.pkl'.")
