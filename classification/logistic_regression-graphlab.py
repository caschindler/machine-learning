# Import graphlab to use methods
from __future__ import division
import graphlab
import math
import string


# DATA PREPARATION

# Example uses baby product reviews on Amazon.com.
products = graphlab.SFrame('amazon_baby.gl/')
products[269]

# Remove punctuation using Python's built-in string functionality.
# and transform the reviews into word-counts.
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

review_without_puctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_puctuation)
products[269]['word_count']

# Aside: For the sake of simplicity, remove all punctuations. 
# A smarter approach to punctuations would preserve phrases such as "I'd", "would've", "hadn't" and so forth.

# Extract sentiments. 
# Assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower to be negative. 
# For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products

# Split data into training and test sets
train_data, test_data = products.random_split(.8, seed=1)


# TRAIN A SENTIMENT CLASSIFIER WITH LOGISTIC REGRESSION USING TRAINING DATA

# Use graphlab function
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
sentiment_model

# Extract the weights and calculate the number of positive and negative weights.
weights = sentiment_model.coefficients
weights.column_names()

num_positive_weights = (weights['value'] > 0).sum()
num_negative_weights = (weights['value'] < 0).sum()
print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights


# MAKE PREDICTIONS WITH THE MODEL

# Make predictions with the model and test data
sample_test_data = test_data[10:13]
sample_test_data
scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores

# Using the scores, make class predictions
simple_class_prediction = scores.apply(lambda x: 1 if x>0 else -1)
print simple_class_prediction

# Calculate the probability that a sentiment is positive.
from math import exp as exp
simple_probabilities = scores.apply(lambda x: 1/(1+exp(-x)))
simple_probabilities
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')

# Find the most positive (and negative) review
import numpy as np
test_prob = sentiment_model.predict(test_data, output_type='probability')
test_data['test_prob'] = test_prob
print test_data.topk('test_prob',k=20).print_rows(num_rows=21)

test_prob_np = test_prob.to_numpy()
test_name_np = test_data['name'].to_numpy()
test_prob_idx_np = test_prob_np.argsort()

positive_20 = test_prob_idx_np[(len(test_prob_idx_np)-20):len(test_prob_idx_np)]
print positive_20
print test_name_np[positive_20]

print test_data.topk('test_prob',k=20, reverse=True).print_rows(num_rows=21)
negative_20 = test_prob_idx_np[:20]
print negative_20
print test_name_np[negative_20]


# EVALUATE THE ACCURACY OF THE MODEL

def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    class_prediction = model.predict(data)
    
    # Compute the number of correctly classified examples
    num_correct = (class_prediction==true_labels).sum()

    # Then compute accuracy by dividing num_correct by total number of examples
    accuracy = num_correct / data.num_rows()
    
    return accuracy

get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])


# LEARN ANOTHER CLASSIFIER WITH FEWER WORDS

# Select words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

# Prepare data
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

# Train model
simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
simple_model

# Check accuracy
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

# Inspect/sort coefficients
simple_model.coefficients
simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)


# COMPARE MODELS

get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
get_classification_accuracy(simple_model, train_data, train_data['sentiment'])

get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])


# VIEW MODEL IN LIGHT OF MAJORITY CLASS CLASSIFIER

# Determine the majority class
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative

# Calculate the accuracy of the majority class
test_num_positive  = (test_data['sentiment'] == +1).sum()
test_num_negative = (test_data['sentiment'] == -1).sum()
print test_num_positive / test_data.num_rows()
print test_num_positive / (test_num_positive + test_num_negative)