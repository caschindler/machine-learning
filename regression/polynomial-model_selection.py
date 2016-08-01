# Load graphlab to be able to use SFrame/SArray
import graphlab
%matplotlib inline
import matplotlib.pyplot as plt


# DATA PREPARATION

# Function to create an SFrame consisting of the powers of an SArray up to a specific degree.
# Takes a SArray/SFrame as feature and a number as degree.
# Returns an SFrame.
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe

# Example runs on a dataset from house sales in King County, the region where the city of Seattle, WA is located.
sales = graphlab.SFrame('kc_house_data.gl/')
# For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living.
# For houses with identical square footage, we break the tie by their prices.
sales = sales.sort(['sqft_living', 'price'])


# EXEMPLIFIED IMPLEMENTATION OF POLYNOMIAL REGRESSION VIA graphlab

# Take degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
# Run regression
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
# Look at coefficients
model1.get("coefficients")
# Visualize
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')

# Take degree 2 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
poly2_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = poly2_features, validation_set = None)
model2.get("coefficients")
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')

# Take degree 3 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.
poly3_data = polynomial_sframe(sales['sqft_living'], 3)
poly3_features = poly3_data.column_names()
poly3_data['price'] = sales['price']
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = poly3_features, validation_set = None)
model3.get("coefficients")
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')

# Take degree 15 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
poly15_features = poly15_data.column_names()
poly15_data['price'] = sales['price']
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = poly15_features, validation_set = None)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')


# SELECT POLYNOMIAL DEGREE FOR REGRESSION

# Function to select a polynomial degree for a regression model with initially one feature.
# Create three datasets: training, validation and testing.
# Create a loop that uses a range of degrees, and for each degree:
# (1) creates an SFrame with feature columns raised to the degrees 1 to xth-Loop (+ target);
# (2) uses the training dataset to train a regression model;
# (3) uses the trained model together with the validation dataset to predict outcomes and calculate RSS for model selection.

(training_and_validation, testing) = sales.random_split(0.9, seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)

for degree in range(1, 15+1):
    train_data = polynomial_sframe(training['sqft_living'], degree)
    train_data_features = train_data.column_names()
    train_data['price'] = training['price']
    trained_model = graphlab.linear_regression.create(train_data, target = 'price', features = train_data_features, validation_set = None, verbose = False)
    valid_data = polynomial_sframe(validation['sqft_living'], degree)
    valid_data['price'] = validation['price']
    valid_yhat = trained_model.predict(valid_data)
    valid_residuals = valid_yhat - valid_data['price']
    valid_rss = (valid_residuals**2).sum()
    print valid_rss

# After model selection (= lowest RSS), use testing data to predict outcomes, calculate RSS and visualize result.
test_data = polynomial_sframe(testing['sqft_living'], 6)
test_data['price'] = testing['price']
test_yhat = trained_model.predict(test_data)
test_residuals = test_yhat - test_data['price']
test_rss = (test_residuals**2).sum()
print test_rss

plt.plot(test_data['power_6'],test_data['price'],'.',
    test_data['power_6'], test_yhat,'-')
