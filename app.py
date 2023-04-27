# importing libraries
from flask import Flask, render_template, request
import random,keywords_database
import requests
from bs4 import BeautifulSoup
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# flask app
app = Flask(__name__)
# using database for chatbot
keywords=keywords_database.keywords
# define a function to generate a response
def generate_response(user_input):
    # tokenize the user's input
    tokens = word_tokenize(user_input.lower())
    # remove stop words and punctuation
    stop_words = set(stopwords.words('english') + list(string.punctuation))

    filtered_tokens = [token for token in tokens if token not in stop_words]
    # check for keywords
    if filtered_tokens:
        print(filtered_tokens)
        for words in keywords:
            if words in filtered_tokens:
                return random.choice(keywords[words])
    for word in keywords:
        if user_input.lower() in word:
            return random.choice(keywords[word])
    # if no keywords match, perform a Google search
    search_url = "https://www.google.com/search?q=" + user_input.replace(" ", "+")
    search_results = requests.get(search_url)
    soup = BeautifulSoup(search_results.text, 'html.parser')
    summary = soup.find_all('div', {'class': 'BNeawe s3v9rd AP7Wnd'})
    # validating the summary
    if len(summary) > 0:
        summary = summary[0].get_text()
        # cleaning the text for summarizing the data.
        summary = re.sub(r'\[[0-9]*\]', '', summary)
        summary = re.sub(r'\s+', ' ', summary)
        return f"I found this information:- {summary.strip()}. Is there anything else I can help you with?"
    else:
        return "I'm sorry, I don't know the answer to that."

# making the chatbot for listening the user input
# while True:
#     user_input = input("You: ")
#     response = generate_response(user_input)
#     print("Chatbot: " + response)
# routing

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # getting the response
        response = generate_response(user_input)
        return render_template('index.html', user_input=user_input, response=response)
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
