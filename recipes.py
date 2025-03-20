


                ### use just for debugging things

import pandas as pd
import os
import ast # used to convert string lists into actual lists

print("CWD = ", os.getcwd())

df = pd.read_csv(r"PP_recipes.csv")  # load preprocessed recipes


# only keeping columns that we need form data set
df = df[['id', 'name_tokens', 'ingredient_tokens']]

# convert it to python list
df['ingredient_tokens'] = df['ingredient_tokens'].apply(ast.literal_eval)

print(df['ingredient_tokens'].iloc[0])
