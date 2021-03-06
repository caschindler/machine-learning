# Import Graphlab to use SArray/SFrame
import graphlab
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


# DATA PREPARATION

# Example runs on a dataset from house sales in King County, the region where the city of Seattle, WA is located.
sales = graphlab.SFrame('kc_house_data.gl/')

# Function to convert SFrame to Numpy array
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)


# MODEL ESTIMATION (RIDGE REGRESSION)

# Create utility function to calculate derivative depending on whether feature is constant or not.
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant == True:
        derivative = 2 * np.dot(errors, feature)
    else:
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight
    return derivative

# Create ridge regression model
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)
    iteration = 0
    while iteration < max_iterations:
        predictions = np.dot(feature_matrix, weights)
        errors = predictions - output
        for i in xrange(len(weights)):
            if i == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, False)
            weights[i] = weights[i] - step_size * derivative
        iteration = iteration + 1
    return weights


# MAKE PREDICTIONS USING SIMPLE LINEAR REGRESSION WITH AND WITHOUT L2 PENALTY

# Feature selection and data preparation
simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

# Create two models: One with no L2 penalty, and one with high penalty.
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
print simple_weights_0_penalty

simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print simple_weights_high_penalty

# Visualization
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')

# Compare and evaluate the quality of models by calculating the sum of squared errors for each model.
print ((predict_output(simple_test_feature_matrix, initial_weights) - test_output)**2).sum()
print ((predict_output(simple_test_feature_matrix, simple_weights_0_penalty) - test_output)**2).sum()
print ((predict_output(simple_test_feature_matrix, simple_weights_high_penalty) - test_output)**2).sum()


# MAKE PREDICTIONS USING MULTIPLE LINEAR REGRESSION WITH AND WITHOUT L2 PENALTY

# Feature selection and data preparation
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

# Create two models: One with no L2 penalty, and one with high penalty.
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
print multiple_weights_0_penalty

multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print multiple_weights_high_penalty

# Visualization
plt.plot(feature_matrix,output,'k.',
         feature_matrix,predict_output(feature_matrix, multiple_weights_0_penalty),'b-',
        feature_matrix,predict_output(feature_matrix, multiple_weights_high_penalty),'r-')

# Compare and evaluate the quality of models by calculating the sum of squared errors for each model.
print ((predict_output(test_feature_matrix, initial_weights) - test_output)**2).sum()
print ((predict_output(test_feature_matrix, multiple_weights_0_penalty) - test_output)**2).sum()
print ((predict_output(test_feature_matrix, multiple_weights_high_penalty) - test_output)**2).sum()

print np.dot(test_feature_matrix[0],multiple_weights_0_penalty)
print np.dot(test_feature_matrix[0],multiple_weights_high_penalty)
print test_output[0]
