# Load graphlab to use methods
import graphlab
graphlab.canvas.set_target('ipynb')


# DATA PREPARATION

# Example runs on Lending Club data
loans = graphlab.SFrame('lending-club-data.gl/')
loans.column_names()
loans['grade'].show()
loans['home_ownership'].show()

# Create the target column by reassigning values from column 'bad loans' (1 => risky and 0 => safe)
# Target column should be 'safe_loans' with values (1 => safe and -1 => risky)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

# Expore the target column 
loans = loans.remove_column('bad_loans')
loans['safe_loans'].show(view = 'Categorical')

# Select features for the classification algorithm
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column for analysis
loans = loans[features + [target]]

# Sample data to balance classes
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)
print "Percentage of safe loans  :", float(len(safe_loans_raw)) / (len(safe_loans_raw) + len(risky_loans_raw))
print "Percentage of risky loans :", float(len(risky_loans_raw)) / (len(safe_loans_raw) + len(risky_loans_raw))

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# Verify new data balance
print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

# Split data into training and validation set
train_data, validation_data = loans_data.random_split(.8, seed=1)


# CREATE DECISION TREE MODEL WITH GRAPHLAB USING TRAINING DATA

# Use decision tree to build a classifier based in training data
decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)

# Visualize the learned model
decision_tree_model.show(view="Tree")


# PREDICT CASES WITH THE DECISION TREE MODEL

# Predict class for selected cases in validation set
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

predicted_class = decision_tree_model.predict(sample_validation_data)
print predicted_class

predicted_class_prob = decision_tree_model.predict(sample_validation_data, output_type='probability')
print predicted_class_prob

# Look at cases also visually
decision_tree_model.show(view="Tree")


# EVALUATE ACCURACY OF THE DECISION TREE MODEL

big_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 10)
print big_model.evaluate(train_data)['accuracy']
print big_model.evaluate(validation_data)['accuracy']

# Quantify the cost of mistakes
class_predictions = decision_tree_model.predict(validation_data)

# False positives
validation_data['class_predictions'] = class_predictions
all_predictions = validation_data[['class_predictions'] + ['safe_loans']]
positive_prediction = all_predictions[all_predictions['class_predictions'] == 1]
false_positive =  positive_prediction[positive_prediction['safe_loans'] ==-1]
print len(positive_prediction)
print len(false_positive)

# False negatives
negative_prediction = all_predictions[all_predictions['class_predictions'] == -1]
false_negative =  negative_prediction[negative_prediction['safe_loans'] ==1]
print len(negative_prediction)
print len(false_negative)