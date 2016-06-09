import graphlab
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)

# ADD DESCRIPTION!

def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)

# Function  which normalizes columns of a given feature matrix.
# The function returns a pair (normalized_features, norms), where the second item contains the norms of original features.
# We will use these norms to normalize the test data in the same way as we normalized the training data.

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

# Single coordinate descent step
# RSS(features) partially derived towards feature_i =
#    = -2 * SUM( feature_i * (output - prediction_without_feature_i) - prediction_only_feature_i )
# ro =      SUM( feature_i * (output - prediction_without_feature_i) )
# Operationalization of ro by means of complete prediction matrix subtraction and adding predicition for feature_i again.

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = np.dot(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = (feature_matrix[:,i] * (output - prediction + weights[i] * feature_matrix[:,i])).sum()

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + (l1_penalty / 2)
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - (l1_penalty / 2)
    else:
        new_weight_i = 0.

    return new_weight_i

# Cyclical coordinate descent

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    converged = False
    weights = np.array(initial_weights)
    counter = 0
    while not converged:
        coordinate_step = 0
        for i in xrange(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            coordinate_step = coordinate_step + abs(weights[i] - old_weights_i)
        if coordinate_step < tolerance:
            converged = True
            print coordinate_step
            print counter
        counter = counter + 1
    return weights

# Evaluating learned models on the test data

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

def compute_rss(feature_matrix, weights, output):
    yhat = np.dot(feature_matrix, weights)
    rss = ((yhat - output)**2).sum()
    return rss
