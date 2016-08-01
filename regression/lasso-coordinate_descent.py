# Load graphlab to be able to use SArray/SFrame
import graphlab
import numpy as np


# DATA PREPARATION

# The example runs on a dataset from house sales in King County,
# the region where the city of Seattle, WA is located.
sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)

# Function to turn SFrame data into numpy array for analysis.
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


# PERFORM COORDINATE DESCENT WITH LASSO

# Function to compute a single coordinate descent step
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

# Function for cyclical coordinate descent
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


# EXAMPLE IMPLEMENTATIONS

# Evaluating LASSO with more features
train_data,test_data = sales.random_split(.8,seed=0)

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']
my_output = 'price'

(train_feature_matrix, train_output) = get_numpy_data(train_data, all_features, my_output)
(train_normalized_feature_matrix, train_norms) = normalize_features(train_feature_matrix)

# Model with l1_penalty of 1e7
initial_weights = np.zeros(len(all_features) + 1)
l1_penalty = 1e7
tolerance = 1.0

weights1e7 = lasso_cyclical_coordinate_descent(train_normalized_feature_matrix, train_output,
                                            initial_weights, l1_penalty, tolerance)
print weights1e7

# Model with higher l1_penalty
initial_weights = np.zeros(len(all_features) + 1)
l1_penalty = 1e8
tolerance = 1.0

weights1e8 = lasso_cyclical_coordinate_descent(train_normalized_feature_matrix, train_output,
                                            initial_weights, l1_penalty, tolerance)
print weights1e8

# Model with lower l1_penalty, but higher tolerance
initial_weights = np.zeros(len(all_features) + 1)
l1_penalty = 1e4
tolerance = 5e5

weights1e4 = lasso_cyclical_coordinate_descent(train_normalized_feature_matrix, train_output,
                                            initial_weights, l1_penalty, tolerance)
print weights1e4

# Rescaling learned weights.
# Recall that we normalized our feature matrix, before learning the weights.
# To use these weights on a test set, we must normalize the test data in the same way.
# We can rescale the learned weights to include the normalization, so we never have to worry about normalizing the test data.
# In this case, we must scale the resulting weights so that we can make predictions with original features.
weights1e4_normalized = weights1e4 / train_norms
weights1e7_normalized = weights1e7 / train_norms
weights1e8_normalized = weights1e8 / train_norms
print weights1e7_normalized[3]


# EVALUATING THE LEARNED MODELS ON TEST DATA

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

def compute_rss(feature_matrix, weights, output):
    yhat = np.dot(feature_matrix, weights)
    rss = ((yhat - output)**2).sum()
    return rss
