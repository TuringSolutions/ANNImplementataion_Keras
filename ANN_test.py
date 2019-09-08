# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:00:30 2019

@author: danish
"""

############## Homework Solution #######################

##################### Testing the model ########################
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np

#loads the model
model = load_model('audiobooks_model.h5')

#Remember we have not given our inputs to the NN as it is. First we have transformed them in the ANN_preprocessing
#We used transform() method from StandardSacler() class of sklearn.preprocessing to normalize all the values.
#Standardize features by removing the mean and scaling to unit variance
""" So it needs to somehow first know about the mean and variance of your data. So fit() or fit_transform() is 
    needed so that StandardScaler can go through all of your data to find the mean and variance. """
sc = StandardScaler()  

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

""" we need to call fit_transform() method only so that test data is scaled in the same way as a training data 
is scaled. And if you'll not use fit_transform() metod then it will also generate an error 
Error: NotFittedError: This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments 
before using this method.
Here fit_transform() is just called to avoid the error and so that transform() method normalize the data same 
according to which our tarin data was normalized."""

_ = sc.fit_transform(np.array([[0,1,2,3,4,5,6,7,8,9]]))

# we have to provide input data to our ANN in a numpy array fashion. During the training the input samples 
# were in horizontal fashion i.e each input was along the horizontal axis (1 rows 10 columns), so we have to
# now this array also must be along the horizontal axis (1 row and 10 columns) otherwise the data will not be
# processed correctly. And for that we will use two square brackets [[values]]. And after this we normalize our
# inputs using transform() method and assign it to scaled_prediction.
scaled_prediction = sc.transform(np.array([[overall_book_len, avg_book_len, ovrall_price, 
                                                   avg_price, review, review_10, min_listened, 
                                                   completion, suprt_req, lvist_pd]]))
                                                   
                                                   
#scaled_prediction = sc.transform(np.array([[1890,3780,10.13,10.13,0,8.91,0.05,81,0,22]]))
scaled_prediction = sc.transform(np.array([[2160,2160,5.33,5.33,0,8.91,0,0,0,1]]))

# Here again we will use model.predict() method and inside this we will provide our scaled or normalized inputs.
single_prediction = model.predict(scaled_prediction)
# single_prediction will contain the probability that whether the customer will convert again or not. So we make
# check if probability is grater than 0.5 then there is a good chance that customer will convert again otherwise not.
single_prediction = (single_prediction > 0.5)
#true means he/she will convert again
if single_prediction:
    print('\nThe customer will convert!')
else:
    print('\nThe customer will not convert!')
    
# So that was it implementation of ANN using keras, next we will cover implemntation of ANN using tensorflow
# The preprocessing code will remain same but after preprocessing we will change things to TF. Building
# a ANN was very easy, but building a ANN with TF takes a littel bit of courage but it will also clear all
# the ambiguities as we will design the layers, batches and weights on our own just by simply using the
# power of TF.