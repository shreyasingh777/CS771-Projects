import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import itertools

######### Generate bigrams from a word
def get_bigrams( word, lim = None ):
  # Get all bigrams
  bg = map( ''.join, list( zip( word, word[1:] ) ) )
  # Remove duplicates and sort them
  bg = sorted( set( bg ) )
  # Make them into an immutable tuple and retain only the first few
  return tuple( bg )[:lim]
####### Processing bigrams
def process_bgs(bigrams):
    bg = sorted(set(bigrams))
    return bg[:5]

def my_fit(words):
    bg_dict = defaultdict(list)
    for word in words:
        bigrams = get_bigrams(word)
        processed_bigrams = process_bgs(bigrams)
        bg_dict[tuple(processed_bigrams)].append(word)
    
    X = []  # Store feat vectors
    y = []  # Store words
    
    # bigram_to_idx is a words that maps each unique bigram to a unique idx.
    bigram_to_idx = {bigram: idx for idx, bigram in enumerate(sorted(set(itertools.chain(*bg_dict.keys()))))}
    
    # Keeping track of all possible words for each set of bigrams
    bigram_to_words = defaultdict(list)
    
    for bigrams, words in bg_dict.items():
        feat_vector = [0] * len(bigram_to_idx)
        for bigram in bigrams:
            feat_vector[bigram_to_idx[bigram]] = 1
        X.append(feat_vector)
        
        # Associate this set of bigrams with all possible words
        for word in words:
            bigram_to_words[tuple(feat_vector)].append(word)
    
    X = np.array(X)
    # Use the first word of each bigram set for training
    y = np.array([words[0] for words in bg_dict.values()])
    
    # Train the decision tree model
    DTC = DecisionTreeClassifier(max_depth=10)
    DTC.fit(X, y)
    
    return [DTC, bigram_to_idx, bigram_to_words]

# Implementing the my_predict() method


def my_predict(model, bigrams):
    processed_bigrams = process_bgs(bigrams)
    feat_vector = [0] * len(model[1])
    for bigram in processed_bigrams:
        if bigram in model[1]:
            feat_vector[model[1][bigram]] = 1
    
    feat_tuple = tuple(feat_vector)

    # Retrieving all possible words that match this set of bigrams
    if feat_tuple in model[2]:
        predictions = model[2][feat_tuple]
    else:
        # using model prediction and look up the closest matches
        prediction = model[0].predict([feat_vector])[0]
        prediction_bigrams = tuple(process_bgs(generate_bigrams(prediction)))
        predictions = bg_dict[prediction_bigrams] if prediction_bigrams in bg_dict else [prediction]
    
    return predictions[:5]