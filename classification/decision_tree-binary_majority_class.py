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
print "Number of features (after binarizing categorical variables) = %s" % len(features)

# Split dataset into train and test set
train_data, test_data = loans_data.random_split(.8, seed=1)


# IMPLEMENT DECISION TREE

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
               
    return leaf 

# Create the decision tree using recursion (and the functions prepared above).
# Takes a SFrame 'data', a list 'features', a string 'target', and integers for 'current_depth' and 'max_depth'.
# Returns a collection of nested dictionaries, each either representing a splitting node with further nested dictionaries,
# or a leaf node created with the create_leaf function.
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node using function intermediate_node_num_mistakes)
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature using function best_splitting_feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth) 

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# CREATE DECISION TREE MODEL USING TRAINING DATA

my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6)


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

# Test case for function
print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])
classify(my_decision_tree, test_data[0], annotate=True)


# EVALUATE ACCURACY OF THE DECISION TREE MODEL

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    mistakes = (prediction != data['safe_loans']).sum()
    classification_error = mistakes / float(len(data))
    return classification_error

evaluate_classification_error(my_decision_tree, test_data)


# VISUALIZE DECISION STUMP

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

print_stump(my_decision_tree)

# Explore the intermediate left subtree
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])

# Explore the left subtree of the left subtree
# First split is on term. 36 months = 1
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])
# First split is on term. 36 months = 0
print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature'])