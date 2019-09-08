# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:28:02 2019

@author: danish
"""

import numpy as np

# We will use the sklearn preprocessing library, as it will be easier to standardize the data.
from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing

# Load the data
raw_csv_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

""" The inputs are all columns in the csv, except for the first one [:,0] (which is just the arbitrary 
    customer IDs that bear no useful information), and the last one [:,-1] (which is our targets). 

    unscaled_inputs_all: it will contain the column number  1-10 which we will be using as input.
    targets_ll: It will contain the targets, which have the information whether the customer is converted
                or not.
                
To achieve the better result we must have to split the data equally i.e both classes should have 
the same ratio in our data, otherwise our model will overfit, if the data of a certain class is 
greater than the other. In  this model we have to classes converted or not converted. So before 
moving on first we have to balance our data.
    
    num_one_targets: It will sum all the ones present in our targets column, in this way we'll know 
                     that our data is balance or not. Our data contains 14084 customers in total, &
                     only 2237 customers have been converted (i.e we have 2237 1's in our data),
                     if we train our model as it is it will overfit the model to not-converted.
                     
How we will balance the data?
We will create a variable that will keep the count number of zeros in our target column, when the value
in this variable will become grater than num_one_targets which is 2237, we will take the indices
of those rows that will have the zero in their targets column, and will start appending these indices
in the list that we have created which is indices_to_remove. This is done with the help of for loop.
After this we'll use np.delete() method to delete the extra rows that contain zero in their target column.
from both inputs and targets also.
    
    unscaled_inputs_equal_priors: Contains the input with equal priors mean that now we'll have the data
                                  which contains equal number of 1's and 0's in there target column.
    targets_equal_priors: same as above."""

unscaled_inputs_all = raw_csv_data[:,1:-1]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:,-1]

# Count how many targets are 1 (meaning that the customer did convert)
num_one_targets = int(np.sum(targets_all))

# Set a counter for targets that are 0 (meaning that the customer did not convert)
zero_targets_counter = 0

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# Declare a variable that will do that:
indices_to_remove = []

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.
# rememeber axis 0, represent row and axis 1 represent column.
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)


"""     scaled_inputs: It'll standardize the inputs by using preprocessing.scale() method. Scaling the 
        data brings all your values onto one scale eliminating the sparsity (In simple words,sparsity 
        means your data is vastly spread out.) and it follows the same concept of Normalization and 
        Standardization.
A little trick is to shuffle the inputs and the target. We are basically keeping the same information but 
in a different order it's possible that the original data set was collected in the order of date. Since 
we will be batching we must shuffle the data. It should be as randomly spread as possible so batching
works fine. Imagine the data is ordered. So each batch represents approximately a different day of purchases. 
Inside the batch the data is homogeneous, while between batches it is very heterogeneous due to promotions 
day of the week effects and so on. This will confuse the stochastic gradient descent when we average the 
loss across batches. So that is why we want to shuffle our data.
        
        shuffled_indices: it will contain an array which will contain continuous numbers from 0 to 4473.
        Then np.random.shuffle() method will shuffle all the values of array.
        
        Then we apply these shuffled indices to the scaled_inputs & targets_equal_priors, which are stored
        in shuffled_inputs & shuffled_targets respectively.
        """

# standardizes the inputs
sc = StandardScaler()        
scaled_inputs = sc.fit_transform(unscaled_inputs_equal_priors)   
     
#scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

#Shuffle the data
# When the data was collected it was actually arranged by date
# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.
# Since we will be batching, we want the data to be as randomly spread out as possible
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

######################### Split the dataset into train, validation, and test ########

""" Now we have to split our datset into training validation and test set. We will split our data by 
    80-10-10 distribution of training, validation, and test set.
    
    samples_count: Contains the total number of inputs, which is 4474.
    
    train_samples_count: 80% of the total data will be used for training. 3579 samples for training.
    
    validation_samples_count: 10% of the total data will be used for validation. 447 samples for validtaion
    
    test_samples_count: All the remaining data which is 10% of total will be used for test. 448 samples for test
    
We have the sizes of the train validation and test next we'll extract them from the big data. which is stored in
shuffled_inputs & shuffled_targets.   
    
    train_inputs: Contains the input (3579 samples of input) data for training. From row 0 to 3579
    
    train_targets: Contains the targets (3579 samples of target) data for training. From row 0 to 3579
    
    validation_inputs: Contains the input (447 samples of input) data for validation. From row 3579 to 4026
    
    validation_targets: Contains the targets (3579 samples of target) data for validation. From row 3579 to 4026
    
    test_inputs: Contains the input (448 samples of input) data for test. From row 4026 to 4474
    
    test_targets: Contains the targets (3579 samples of target) data for test. From row 4026 to 4474
    """

# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print('Number of ones: '+'{0:.1f}.'.format(np.sum(train_targets)), 
      'Total Samples for training: '+'{0}. '.format(train_samples_count), 
      'Split of training data: '+'{0:.3f}.'.format(np.sum(train_targets) / train_samples_count))

print('Number of ones: '+'{0:.1f}.'.format(np.sum(validation_targets)), 
      'Total Samples for validation: '+'{0}. '.format(validation_samples_count), 
      'Split of validation data: '+'{0:.3f}.'.format(np.sum(validation_targets) / validation_samples_count))

print('Number of ones: '+'{0:.1f}.'.format(np.sum(test_targets)), 
      'Total Samples for test: '+'{0}. '.format(test_samples_count), 
      'Split of test data: '+'{0:.3f}.'.format(np.sum(test_targets) / test_samples_count))



####################  Save the three datasets in *.npz ##################
# Save the three datasets in *.npz.
# np.savez() Save several arrays into a single file in uncompressed .npz format.
# it is extremely valuable to name them in such a coherent way we will see when we will write our class that will split 
# the data into batches. Saving the preroprocessing that we have done on our data is also valuable becuase we don't 
# have to re-run this section of code .npz format is tensorflow friendly format. 

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)










