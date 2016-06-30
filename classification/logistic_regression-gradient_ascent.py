# Load Graphlab to use SFrames/SArrays
import graphlab

# DATA PREPARATION

# Example runs on an Amazon baby products review dataset
products = graphlab.SFrame('amazon_baby_subset.gl/')

# Check balance of positive and negative reviews
print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

# Perform some simple feature cleaning using SFrames.
# For simplicity in the current example, do not use bag-of-words features, but only 193 words (for simplicity).
# Use a compiled list of 193 most frequent words from a JSON file.
import json
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

# Now, perform 2 data transformations:
# 1. Remove punctuation using Python's built-in string functionality.
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)
products['review_clean'] = products['review'].apply(remove_punctuation)
# 2. Compute word counts (only for important_words)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

# Example: Compute the number of product reviews that contain the word perfect.
contains_perfect = products['perfect'] >= +1
print contains_perfect.sum()

# Convert SFrame to Numpy array
import numpy as np
def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
feature_matrix.shape

# MODEL IMPLEMENTATION (LOGISTIC REGRESSION WITH GRADIENT ASCENT)

# Estimate conditional probability with link function
# Produce probablistic estimate for P(y_i = +1 | x_i, w) and estimate ranges between 0 and 1.
def predict_probability(feature_matrix, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    predictions = 1 / (1 + np.exp(-scores))
    return predictions

# Compute the log likelihood for the entire dataset
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    lp = np.sum((indicator-1)*scores - logexp)
    return lp

# Create logistic regression model with gradient descent
from math import sqrt
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Compute the errors between indicator value for (y_i = +1) and prediction P(y_i = +1|x_i,w)
        scores = np.dot(feature_matrix, coefficients)
        predictions = 1 / (1 + np.exp(-scores))
        indicator = (sentiment==+1)
        errors = indicator - predictions
  	    # loop over each coefficient
        for j in xrange(len(coefficients)):
            # Compute the derivative for coefficient[j].
            derivative = np.dot(errors, feature_matrix[:,j])
            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative
        # Check whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

# MAKE PREDICTIONS WITH THE BUILT MODEL AND EVALUATE MODEL QUALITY

# Estimate coefficients and predict sentiment
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194), step_size=1e-7, max_iter=301)
scores = np.dot(feature_matrix, coefficients)
class_prediction = scores / np.abs(scores)
print (class_prediction == 1).sum()
print (class_prediction == -1).sum()
print len(class_prediction)

# Evaluate accuracy
num_mistakes = (sentiment != class_prediction).sum()
num_correct = len(products) - num_mistakes
accuracy = float(num_correct) / len(products)
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

# Identify words that contribute most to positive and negative sentiment
# Treat each coefficient as a tuple, i.e. (word, coefficient_value).
# Sort all the (word, coefficient_value) tuples by coefficient_value in descending order.
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
# 10 most positive
print word_coefficient_tuples[:10]
# 10 most negative (two lines due to Python indexing)
print word_coefficient_tuples[-10:-1]
print word_coefficient_tuples[-1]
