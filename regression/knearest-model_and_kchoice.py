# Import graphlab to be able to use SFrame/SArray

import graphlab
import numpy as np


# DATA PREPARATION

# The example runs on a dataset from house sales in King County,
# the region where the city of Seattle, WA is located.
sales = graphlab.SFrame('kc_house_data_small.gl/')

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

# Split data into training, validation and test set.
(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)

# Extract features and normalize
feature_list = ['bedrooms',
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
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']

features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms


# PERFORM 1-NEAREST NEIGHBOR REGRESSION

# Function to calculate distance from one query house to all training houses.
def euclid_distance(feature_matrix, query_vector):
    diff = feature_matrix[0:len(feature_matrix)] - query_vector
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances

nearest_neighbour_distance = euclid_distance(features_train, features_test[2])
print min(nearest_neighbour_distance)
min_distance_index, min_distance = min(enumerate(nearest_neighbour_distance), key=lambda p: p[1])
print min_distance_index, min_distance


# PERFORM K-NEAREST NEIGHBOR REGRESSION

# Fetch k-nearest neighbours.
def k_nearest_neighbors(k, feature_matrix, query_vector):
    diff = feature_matrix[0:len(feature_matrix)] - query_vector
    distances = np.sqrt(np.sum(diff**2, axis=1))
    k_nearest = np.argpartition(distances, k)[:k]
    return k_nearest

# Note: np.argpartition guarantees that the kth element is in sorted position and all smaller elements will be moved before it.
# ...Thus the first k elements will be the k-smallest elements.
# ...It does not sort the entire array!
k_nearest_neighbors(4, features_train, features_test[2])

# Make a prediction on a single query.
def predict_output(k, feature_matrix, output, query_vector):
    k_nearest = k_nearest_neighbors(k, feature_matrix, query_vector)
    yhat = output[k_nearest].mean()
    return yhat

predict_output(4, features_train, output_train, features_test[2])

# Make predictions on multiple queries.
def predict_output(k, feature_matrix, output, query_feature_matrix):
    yhat_list = []
    for i in xrange(len(query_feature_matrix)):
        k_nearest = k_nearest_neighbors(k, feature_matrix, query_feature_matrix[i])
        yhat = output[k_nearest].mean()
        yhat_list.append(yhat)
    return yhat_list

predict_output(10, features_train, output_train, features_test[0:10])

# Choosing the best value of k using a validation set
rss_all = []
for k in xrange(1,16):
    yhat = predict_output(k, features_train, output_train, features_valid)
    rss_all.append(((yhat - output_valid)**2).sum())
print rss_all

# To visualize the performance as a function of k, plot the RSS on the VALIDATION set for each considered k value:
%matplotlib inline
import matplotlib.pyplot as plt

kvals = range(1, 16)
plt.plot(kvals, rss_all,'bo-')
