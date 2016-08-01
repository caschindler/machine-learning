# Load graphlab to be able to use SFrames/SArrays
import graphlab


# DATA PREPARATION

# Example runs on Lending Club data
loans = graphlab.SFrame('lending-club-data.gl/')

# Create the target column by reassigning values from column 'bad loans' (1 => risky and 0 => safe)
# Target column should be 'safe_loans' with values (1 => safe and -1 => risky)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

# Select features for the classification algorithm
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]

# Subsample dataset to make sure classes are balanced
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

# Transform categorical into binary features
loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features

# Split dataset into train and test set
train_data, validation_set = loans_data.random_split(.8, seed=1)


# IMPLEMENT DECISION TREE

# Early stopping condition 2: Minimum node size
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size:
        return True

# Early stopping condition 3: Minimum gain in error reduction
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    error_reduction = error_before_split - error_after_split
    return error_reduction

# Create function to count number of mistakes while predicting majority class
# Takes a SArray 'labels_in_node'.
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0

    # Count the number of 1's (safe loans)
    num_ones = len(labels_in_node[labels_in_node == 1])

    # Count the number of -1's (risky loans)
    num_minus_ones = len(labels_in_node[labels_in_node == -1])

    # Return the number of mistakes that the majority classifier makes.
    if num_ones > num_minus_ones:
        return num_minus_ones
    else:
        return num_ones

# Create function to pick best feature to split on
# Takes a SFrame 'data', a list 'features' and a string 'target'. Returns a string.
def best_splitting_feature(data, features, target):

    target_values = data[target]
    best_feature = None # Keep track of the best feature
    best_error = 10     # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]

        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]

        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        # Compute the classification error of this split.
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error

    return best_feature

# Create a function to create a leaf node
# Takes a SArray 'target_values'. Returns a dictionary.
def create_leaf(target_values):

    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # Return the leaf node
    return leaf

# Create the decision tree using recursion (and the functions prepared above).
# Takes a SFrame 'data', a list 'features', a string 'target', and integers for 'current_depth' and 'max_depth'.
# Returns a collection of nested dictionaries, each either representing a splitting node with further nested dictionaries,
# or a leaf node created with the create_leaf function.
def decision_tree_create(data, features, target, current_depth = 0,
                         max_depth = 10, min_node_size=1,
                         min_error_reduction=0.0):

    remaining_features = features[:] # Make a copy of the features.

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))


    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if len(target_values) <= min_node_size:
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Perfect split. Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Perfect split. Creating leaf node."
        return create_leaf(right_split[target])


    # Remove feature that was used to split.
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))


    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)

    right_tree = decision_tree_create(right_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)


    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}

# Function to count the nodes in the tree
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# CREATE DECISION TREE MODEL USING TRAINING DATA
my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 100, min_error_reduction=0.0)


# MAKE PREDICTIONS WITH THE BUILT DECISION TREE

# Create a function that decides on a direction at a node in a decision tree
# based on value of an input case relating to the splitting feature at that node.
# The function calls itself again as long as no leaf is reached, thus travelling through the tree.
# Takes a decision tree model 'tree' made from nested dictionaries as above
# and a row from a dataset 'x' as case to be tested, e.g.'dataset[0]'.
# Returns a final class prediction integer (1 or -1).
def classify(tree, x, annotate = False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

# Test case
print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_set[0])


# EVALUATE ACCURACY OF THE DECISION TREE MODEL

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))

    # Once you've made the predictions, calculate the classification error and return it
    mistakes = (prediction != data['safe_loans']).sum()
    classification_error = mistakes / float(len(data))
    return classification_error

evaluate_classification_error(my_decision_tree_new, validation_set)


# EXPLORING THE EFFECT OF max_depth (to shallow, just about right, probably too deep)

model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2,
                                        min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14,
                                        min_node_size = 0, min_error_reduction=-1)

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data)

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, validation_set)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, validation_set)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, validation_set)


# MEASURING THE COMPLEXITY OF A TREE

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

print count_leaves(model_1)
print count_leaves(model_2)
print count_leaves(model_3)


# EXPLORING THE EFFECT OF min_error_reduction (negative, just about right, too positive)

model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 0, min_error_reduction=5)

print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set)

print count_leaves(model_4)
print count_leaves(model_5)
print count_leaves(model_6)


# EXPLORING THE EFFECT OF min_node_size (too small, just about right, too large)

model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                        min_node_size = 50000, min_error_reduction=-1)

print "Training data, classification error (model 7):", evaluate_classification_error(model_7, train_data)
print "Training data, classification error (model 8):", evaluate_classification_error(model_8, train_data)
print "Training data, classification error (model 9):", evaluate_classification_error(model_9, train_data)

print count_leaves(model_7)
print count_leaves(model_8)
print count_leaves(model_9)
