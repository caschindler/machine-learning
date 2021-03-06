# Load graphlab to be able to use SFrame/SArray
import graphlab
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

# DATA PREPARATION

# Function to produce an SFrame with columns containing the powers of a given input.
def polynomial_sframe(feature, degree):
    poly_sframe = graphlab.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe

# Example runs on a dataset from house sales in King County, the region where the city of Seattle, WA is located.
sales = graphlab.SFrame('kc_house_data.gl/')
# For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living.
# For houses with identical square footage, we break the tie by their prices.
sales = sales.sort(['sqft_living','price'])
# Split the data
(train_valid, test) = sales.random_split(.9, seed=1)
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments.
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


# SELECTING AN L2 PENALTY VIA CROSS-VALIDATION

# We will implement a kind of cross-validation called k-fold cross-validation.
# The method gets its name because it involves dividing the training set into k segments of roughtly equal size.
# Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set.
# The major difference is that we repeat the process k times as follows:
# Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# ...
# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

# Function to compute the average validation error for a model
# Takes a value k for dataset segmentation, a value for l2 penalty, an SFrame dataset, a string for output variable, and a list with features.
# Trains k models with resliced train data, and for each returns the error in predicting with the complementary validation data.
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    sum_validation_rss = 0
    for i in xrange(k):
        validation_data_start = (n*i)/k
        validation_data_end = (n*(i+1))/k-1
        validation = data[validation_data_start:validation_data_end]
        training = data[validation_data_end+1:n].append(data[0:validation_data_start])
        trained_model = graphlab.linear_regression.create(training, target = output_name, features = features_list, l2_penalty=l2_penalty, validation_set=None, verbose = False)
        validation_yhat = trained_model.predict(validation)
        validation_rss = ((validation[output_name] - validation_yhat)**2).sum()
        sum_validation_rss = sum_validation_rss + validation_rss
    av_validation_rss = sum_validation_rss / k
    return av_validation_rss

# Loop to find the model that minimizes the average validation error (example: 15th degree polynomial).
poly15 = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
poly15_features = poly15.column_names()
poly15['price'] = train_valid_shuffled['price']

l2_penalty_list = np.logspace(1, 7, num=13)
av_validation_rss = []

for l2_penalty in l2_penalty_list:
    av_validation_rss.append(k_fold_cross_validation(10, l2_penalty, poly15, 'price', poly15_features))
print av_validation_rss

# Plot to make results more intuitive.
plt.plot(l2_penalty_list, av_validation_rss, '.')
plt.xscale('log')
plt.title('average validation error')
plt.grid(True)

# Once you found the best value for the L2 penalty by cross-validating,
# use this L2-value to retrain a final model with all of the training data (entire dataset except for the test part).
poly15_model = graphlab.linear_regression.create(poly15, target = 'price', features = poly15_features, l2_penalty=1000, validation_set=None)

# Use the learned model to predict outcomes with the test data and calculate the test error.
test_p15 = polynomial_sframe(test['sqft_living'], 15)
test_p15_features = test_p15.column_names()
test_p15['price'] = test['price']
test_yhat = poly15_model.predict(test_p15)
test_rss = ((test_p15['price'] - test_yhat)**2).sum()
print test_rss
