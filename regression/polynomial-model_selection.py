import graphlab
%matplotlib inline
import matplotlib.pyplot as plt

# Function to create an SFrame consisting of the powers of an SArray up to a specific degree.
# Takes a SArray/SFrame as feature and a number as degree.
# Returns an SFrame.

def polynomial_sframe(feature, degree):
    poly_sframe = graphlab.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe

# Selecting a polynomial degree for a regression model with initially one feature.
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
