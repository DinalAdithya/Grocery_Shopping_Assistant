import random
import json
import torch
import pickle
from model import NeuralNet
from my_nltk import bag_of_words, tokenize
from recipes_model import df

print(df.columns.tolist())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
print("Model loaded successfully:", model)

bot_name = "Natsu"


def get_chat_response(msg):


    """sentence = tokenize(msg)
    print(f"Tokenized Sentence: {sentence}")

    X = bag_of_words(sentence, all_words)
    print(f"Bag of Words Vector: {X}")"""



    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.70:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand...."


# load the saved model
with open("recipe_model.pkl", "rb") as f:
    recipe_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# function to get recipe recommendation
def recommend_recipe(user_ingredients):
    print(f"User ingredients: {user_ingredients}")  # Debugging
    # convert user input to a vector
    user_vector = vectorizer.transform([' '.join(user_ingredients)])

    # find the closest recipe
    _, indices = recipe_model.kneighbors(user_vector)

    # print("Available columns in DataFrame:", df.columns)

    recommended_recipes = df.iloc[indices[0]]['Title'].tolist()

    print(f"recommended_recipes : {recommended_recipes}")  # debug code

    return recommended_recipes





conversation_context = {
        "last_selected_recipe": None
    }

# Example chatbot function using the model
def get_recipe_recommendation(user_input):

    user_input = user_input.lower()

    # Normalize recipe titles for easier matching
    recipe_titles = [title.lower() for title in df['Title'].tolist()]

    # if user ask for recipe explicitly
    if "suggest a recipe" in user_input or "recipe" in user_input:
        ingredients = user_input.replace("suggest a recipe with ", "").split(", ")
        recommended = recommend_recipe(ingredients)
        if recommended:
            # save the first suggestion
            conversation_context["last_selected_recipe"] = recommended[0]
            return f"Here are some recipe ideas: {recommended}"
        else:
            return f"Sorry I couldn't find my recipes with those ingredients."

    # if user input is a list of ingredients
    if "," in user_input or user_input in ['egg', 'milk', 'butter']:
        ingredients = [item.strip() for item in user_input.split(",") if item.strip()]
        recommended = recommend_recipe(ingredients)
        if recommended:
            # save the first suggestion
            conversation_context["last_selected_recipe"] = recommended[0]
            return f"Here are some recipe ideas: {recommended}"
        else:
            return f"Sorry I couldn't find my recipes with those ingredients."

    # If user in put matches recipe name(token)
    if user_input in recipe_titles:
        conversation_context["last_selected_recipe"] = user_input
        return f"Great choice! Would you like the full recipe for '{user_input.title()}'?"

    # if user said yes and recipe was selected earlier
    if user_input in ["yes", " yeah", "sure", "yup"]:
        recipe = conversation_context.get("last_selected_recipe")
        if recipe:
            recipe_details = df[df['Title'].str.lower() == recipe.lower()]

            if not recipe_details.empty:
                ingredients = recipe_details.iloc[0]['Cleaned_Ingredients']
                instructions = recipe_details.iloc[0]['Instructions']
                return (
                    f"Here's the full recipe for **{recipe.title()}**:\n\n"
                    f"**Ingredients:**\n{ingredients}\n\n"
                    f"**Instructions:**\n{instructions}"
                )
            else:
                return "Sorry, I culdn't find that recipe"
        else:
            return "I'm not sure witch recipe u ment. can u repeat the name?"

    # otherwise use intent base chatbot -> "I dont Understand"
    return get_chat_response(user_input)


## to see all the tags are loaded or not
print([intent["tag"] for intent in intents["intents"]])

if __name__ == "__main__":
    print("Let's chat! type 'quit' to exit")
    while True:
        sentence = input('me: ')
        if sentence == "quit":
            break

        resp = get_recipe_recommendation(sentence)
        print(resp)
