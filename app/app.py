import torch
import json
import torch.nn as nn
import pickle

from flask import Flask, render_template, request
from fuzzywuzzy import process
from numpy import dot
from numpy.linalg import norm
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from class_function import Skipgram, SkipgramNeg, Glove

app = Flask(__name__)



# Importing training data
Data = pickle.load(open('../models/Data.pkl', 'rb'))

corpus = Data['corpus']
vocab = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['embedding_size']
window_size = Data['window_size']

# Cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Function to correct misspelled words
def correct_spelling(word, vocab, threshold=80):

    # Find the best match in the vocabulary
    match, score = process.extractOne(word, vocab)
    
    # Return the corrected word if the score is above the threshold
    if score >= threshold:
        return match
    else:
        return word  # Return the original word if no good match is found

# Function to get the top 10 similar words
def get_top_similar_words(model, word_input):

    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            # Correct the spelling of the input word
            corrected_word = correct_spelling(word_input, vocab)
            
            # If the corrected word is different, notify the user
            if corrected_word != word_input:
                print(f"Corrected '{word_input}' to '{corrected_word}'")
            
            # Get the word embedding of the corrected word
            word_embed = model.get_vector(corrected_word).detach().numpy().flatten()

            similarity_dict = {}

            # Compute cosine similarity for each word in the vocabulary
            for a in vocab:
                try:
                    a_embed = model.get_vector(a).detach().numpy().flatten()
                    similarity_dict[a] = cos_sim(word_embed, a_embed)
                except KeyError:
                    continue  # Skip words not in the model's vocabulary

            # Sort the dictionary by similarity in descending order
            similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

            # Return top 10 similar words
            return [f"{i+1}. {similarity_dict_sorted[i][0]} ({similarity_dict_sorted[i][1]:.8f})" for i in range(10)]

        else:
            return ["The system can search with 1 word only."]

    except KeyError:
        return ["The word is not in my corpus. Please enter a new word."]


@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    glove_output = []
    gensim_output = []
    skipgram_output = []
    skipgram_neg_output = []

    if request.method == 'POST':
        search_query = request.form['search_query']

        # Import the saved Skipgram model
        skipgram = Skipgram(voc_size, embed_size)
        skipgram.load_state_dict(torch.load('../models/Word2Vec(Skipgram).pt'),  strict=False)
        skipgram.eval()

        # Import the saved negative Skipgram model
        state_dict = torch.load('../models/Word2Vec(Neg_Sampling).pt')
        skipgramNeg = SkipgramNeg(voc_size, embed_size)
        skipgramNeg.load_state_dict(state_dict)
        skipgramNeg.eval()

        # Import the saved Glove from scratch model
        glove = Glove(voc_size, embed_size)
        glove.load_state_dict(torch.load('../models/Glove_from_scratch.pt'),  strict=False)
        glove.eval()

        # Load Gensim model
        glove_file = datapath('glove.6B.100d.txt')
        gensim_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

        # List of models to loop through
        models = [("Glove", glove), ("Skipgram", skipgram), ("SkipgramNeg", skipgramNeg)]
        
        model_outputs = {}

        for model_name, model in models:
            try:
                model_outputs[model_name] = get_top_similar_words(model, search_query)
            except KeyError as e:
                model_outputs[model_name] = [str(e)]  # Handle unknown words
        
        # Assign outputs to respective variables
        glove_output = model_outputs.get("Glove", ["No results available."])
        skipgram_output = model_outputs.get("Skipgram", ["No results available."])
        skipgram_neg_output = model_outputs.get("SkipgramNeg", ["No results available."])
        gensim_output = gensim_model.most_similar(search_query, topn=10)
        gensim_output = [f"{i+1}. {word} ({similarity:.4f})" for i, (word, similarity) in enumerate(gensim_output)]

    return render_template('index.html', search_query=search_query, 
                           glove_output=glove_output, gensim_output=gensim_output, 
                           skipgram_output=skipgram_output, skipgram_neg_output=skipgram_neg_output)
if __name__ == '__main__':
    app.run(debug=True)
