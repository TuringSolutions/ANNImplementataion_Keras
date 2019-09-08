# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:28:35 2019

@author: danish
"""

import numpy as np

""" Keras is compact, easy to learn, high-level Python library run on top of TensorFlow framework. 
    It is made with focus of understanding deep learning techniques, such as creating layers for 
    neural networks maintaining the concepts of shapes and mathematical details. The creation of
    framework have two types:
    1. Sequential API
    2. Functional API
    We are going to use sequential API, because functional API is somehow identical to the low level libray 
    which is tensorflow, and we are going to design our ANN in Tensorflow as well. Developing a model in
    keras involves 8 steps.
    1.Loading the data  2.Preprocess the loaded data  3.Definition of model  4.Compiling the model
    5.Fit the specified model  6.Evaluate it  7.Save the model  8.Make the required predictions  
    We've already performed the step 1 & 2 in ANN_preprocessing.py. This will take care of step 3-step 7.""" 
import keras
from keras.models import Sequential
from keras.layers import Dense
#from time import time
#from tensorflow.python.keras.callbacks import TensorBoard
#import tensorflow as tf


# Importing the dataset
# The dataset that loads is one of "train", "validation", "test".
# e.g. if I call this class with x('train',5), it will load 'Audiobooks_data_train.npz'
dataset = 'train'
npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))

train_inputs, train_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

dataset = 'test'
npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
test_targets = test_targets == 1



######################### Step 3: Definition of Model #########################
""" In this step we will intialize and define our model.  The Sequential model is a linear stack of layers.
    One can create a Sequential model by passing a list of layer instances to the constructor or can create
    an object of Sequential() class. What we are going to do is, we are going to create an object of 
    Sequential() class. Then we will add layers one by one. The layer are added to the model using add() 
    function from keras. We have to build a fully connected neural network and we know that every layer of a
    neural network have certain pramaeters such as number of neurons or nodes, the activation function and
    the weights for neurons. 
    
    The input layer and first hidden layer in keras are created using single keras line, as we know that input 
    layer just simply pass on the values just by taking the sum of product of weights and then the activation
    function is applied on 1st hidden layer. The model needs to know what input shape it should expect. For
    this reason, the first layer in a Sequential model (and only the first, because following layers can do
    automatic shape inference) needs to receive information about its input shape. So we have to define the 
    input dimensions and number of nodes of first hidden layer in the same line. And for that we will be 
    using Dense() function which will take care of number of neurons, intialization of weights and activation 
    function. So the firts layer we will with the help of Dense() function and it will take 4 argumnets and
    for all other layers, Dense() function will take 3 arguments. 
    
    units: The number of neurons/nodes in a layer. Generally there is no specifc method to define the number 
           of neurons. This parameter varies with the complexity of the problem, but there are two practices
           that are used to define the number of neurons.
           1. Taking the average of input & output dimensions. 2. Parameter Tuning (Experimenting with 
                                                                  different parameters of the model.)
           We are going to use the first practice, we have 10 input dimensions (which is the number of our
           independent variables) and as have binary outcome/output. So we will have only 1 output unit. So the
           average will be 10+1/2 = 5.5, so we are going to use 6 units/neurons for hidden layer.
           
    kernel_initializer: Take care of intialization of weights. We have to randomly intialize the weights using
                        a uniform distribution and the weights must be close to zero but not 0. And for that we
                        will use 'uniform' distribution that will intialize the weights between 0 & 1.
                        
    activation: Take care of activation function, which will be applied to the set of inputs (the sum of product
                of weights and inputs). For the input layers we are going to use 'relu'(Rectifier Linear Unit)
                because it is the best one for input layers based on the research and experimentation. And for
                output layer we will use 'sigmoid activation function'. The sigmoid activation function allow to 
                get the probabailities of different classes, in our case it will tell us what is the probability 
                that the customer will convert and what will be the probability that the customer will not convert.
                
    input_dim:  number of input dimensions/nodes. As this is the first layer.
    """


# Initialising the ANN
model = Sequential()
# First hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
# Second hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#tf.enable_eager_execution()
#Generating Data for tensorboard
#tensorboard = TensorBoard(log_dir='.\logs')

######################### Step 4: Compiling the model #########################
""" Before training a model, you need to configure the learning process, which is done via the compile method.
    As we have created our neural network now in this step we'll apply gradient descent on it with the help
    of compile() method. It receives three arguments:
    
        optimizer: Optimizers shape and mold your model into its most accurate possible form by futzing with
                   the weights. With the help of optimzier we find optimal set of weights for our NN. Here in
                   this argument we define the stochastic gradient descent algorithm (SGD), there are several 
                   types of SGD algorithms, and a very efficient algorithm is 'adam'. That we will provide to
                   this optimzer argument.
        
        loss: This argument refers to the loss function that will calculated between the actual/label value and
              predicted value. The loss function is within the SGD algorithm that is within the adam algo. Because
              SGD is based on a loss function that we need to optimize to find optimal weights. 
        
        metrics: A criterion to evaluate the model. When the weights are updated after a batch of observation is 
                 completed, the algorithm uses the accuracy criterion to improve the performance of model. """

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

####################### Step 5: Fit the specified model #######################
""" In this step we will train our ANN. Keras models are trained on Numpy arrays of input data and labels(targets). 
    For training a model, we will typically use the fit function. it takes the following 4 arguments:
    
    train_inputs: The input in the form of numpy array which are our features.
        
    train_targets: The labeled output(expected values), on the basis of which loss is computed.
        
    batch_size: The number of observations that will be feed to the ANN, and after each batch NN update its
                weights with the help of SGD algo. For example if we have 3580 input samples and we set batch
                size to 10. Then the number of batches will be 3580/10 = 358 batches. The weights are updated
                when one batch of observation is passed into the NN.
        
    epochs: When whole set of observations or all the batches are passed into the NN one epoch is completed
            or in simple terms when one iteration is completed it makes the one epoch"""

# Fitting the ANN to the Training set
model.fit(train_inputs, train_targets, batch_size = 10, epochs = 100)
#model.fit(train_inputs, train_targets, batch_size = 10, epochs = 100, callbacks = [tensorboard])
print('\n\nThe tarining has been completed: ')


######################### Step 6: Evaluating the model ########################
""" In this section we will evaluate the performance or accuracy of our model on the basis of the test set
    that we've have created. But the question is why the evaluation is necessary? The purpose of evaluation 
    is to test a model on different data than it was trained on. This provides an unbiased estimate of 
    learning performance. So for that we have our test set which includes {test_inputs & test_targets} on the
    basis of test set we will predict the accuracy of our model. For that first we have to calculate the 
    probablities of the test_input (whether the customers in test will convert or not). We will consider that 
    a customer will convert if its probability is greater than 0.5 and the customer will not convert if its
    probability is less than 0.5. Then we will compare these results with the test_targets to calculate the
    accuracy of our model on a unseen dataset.
    target_prediction: A list that will hold the status of each customer whether the customer will convert again
                       or will not.
                
    count: Holds the number of customer which have the chance to convert again.
    
    target_pred = Holds the probabilities for each of the customer which is present in test set. The probabilities
                  are calculated by using predict() method. When we use predict() method ANN just forward
                  propogates and tell the results.
    target_pred: Holds the binary result for a customer. We make a threshold on the probabilities of the 
                 customer that if a certain customer have probability greater than 0.5 say it True(customer
                 will convert) otherwise False(customer will not convert). We have to convert these predictions
                 into binary because our test_targets are also in binary and we have to compare these two
                 parameters (target_pred & test_targets) to calculate the accuracy."""

# Predicting the Test set results
target_prediction= []
count=0
target_pred = model.predict(test_inputs)
target_pred = target_pred > 0.5
for i in range(len(target_pred)):
    if target_pred[i]:
        target_prediction.append('convert')
        count +=1
    else:
        target_prediction.append('will not convert')
print('\nOut of {0} customers'.format(len(target_pred))+
      ' There is a chance that {0} customers will convert'.format(count)+
      ' & {0} customers will not convert!'.format(len(target_pred)-count))


print('\nEvaluating test accuracy!')

# Making the Coomparison between test_targets(expected value) and target_pred(predicted value)
""" Here we will calculate the accuracy. We will loop on test_targets(expected value) and target_pred 
    (predicted value) and match them, the values we will match will have the same indexes for example
    we will match test_targets[0] & test_targets[0] if both are True or both or False then we'll increment
    the correct_pred otherwise wrong_pred will be incremented. In this way we'll calculate the total 
    number of correct predictions that our model has predicted. Then we will calculate the test_accuracy
    by following formula
    test_accuracy = Total Number of correct predictions/Total number of test samples * 100 
    There is another method to calculate the accuracy by using confusion_matrix() from sklearn.metrics.
    This method returns a 2 by  2 array, which contains the values of coorect and wrong predictions. First
    and fourth index contains values of correct predictions while 2nd & 3rd index contains the values of
    wrong predictions and from total number of correct predictions we can calculate accuracy as described
    above."""

correct_pred = 0
wrong_pred = 0
for pred in range(len(test_targets)):
    if test_targets[pred] and target_pred[pred] or (target_pred[pred] == False and test_targets[pred] == False):
        correct_pred += 1
    else:
        wrong_pred += 1
test_accuracy = correct_pred/len(test_targets)*100
print('\nTest accuracy: '+'{0:.3f}'.format(test_accuracy))

########################### Step 7: Save the Model ############################
""" It is one of the best practices in deep learning to save your model so you can reuse it whenever you want
    without going through the complete process of preprocessing and training of model. We can use model.save(filepath) 
    to save a Keras model into a single HDF5 file which will contain:
        1. the architecture of the model, allowing to re-create the model
        2. the weights of the model
        3. the training configuration (loss, optimizer)
        4. the state of the optimizer, allowing to resume training exactly where you left off.
    And then you can use keras.models.load_model(filepath) to reinstantiate your model. load_model(filepath) 
    will also take care of compiling the model using the saved training configuration (unless the model was 
    never compiled in the first place).
    To save the keras model you just need to call the save() method using the object that we have created for
    our keras model."""
#Saving the model
model.save('audiobooks_model.h5')  # creates a HDF5 file 'audiobooks_model.h5'
print('\nThe model has been saved!')

############################### Homework ######################################
#Load the saved model. Then you have to write the code to predict for a single customer that whether
#he/she will convert again or not. The data is given below.
overall_book_len = 648 #overall book length.
avg_book_len = 648 #average book length.
ovrall_price = 5.33 #overall price.
avg_price = 5.33 #average price.
review = 0 #reviewd the product or not.
review_10 = 10 #review out of 10.
min_listened = 0.27 # total minutes listened.
completion = 583.2 # Completion, how much of book is completed.
suprt_req = 0 # Support requests.
lvist_pd = 366 #difference  = (last visted - purchase date)