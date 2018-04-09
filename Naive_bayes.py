
# coding: utf-8

# In[1]:

#Python program for Naive-Bayse algorithm with laplacian smoothing
import csv
import random
import math
import numpy as np
from collections import Counter, defaultdict


# In[2]:


# Function to obtain training data from the csv file.
def input():
    datain = []
    dataout = []
    with open('Q2-tennis.csv') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                j=0

                for row in reader:
                    temp1 = 0
                    temp2 = 0
                    temp3 = 0
                    temp4 = 0
                    for i in range(5):
                        if (i==0):
                            if (row[i]== "sunny"):
                                temp1 = 0
                            elif(row[i]== "overcast"):
                                temp1 = 1    
                            else:
                                temp1 = 2 
                        elif (i==1):
                            if (row[i]== "hot"):
                                temp2 = 0
                            elif(row[i]== "mild"):
                                temp2 = 1 
                            else:
                                temp2 = 2
                        elif (i==2):
                            if (row[i]== "high"):
                                temp3 = 0
                            else:
                                temp3 = 1  
                        elif (i==3):
                            if(row[i]=="false"):
                                temp4 = 0
                            else:
                                temp4 = 1
                        else:
                            if(row[i]=="yes"):
                                dataout.append(1)
                            else:
                                dataout.append(0)
                    datain.append([temp1,temp2,temp3,temp4])
                    j = j+1
                        #datain[j][i]= temp
    training = np.asarray(datain)     
    outcome = np.asarray(dataout)
    return training,outcome


# In[3]:

#Function for calculating the prior probabilities
def occurrences(outcome):
    no_of_examples = len(outcome)
    prob = dict(Counter(outcome))
    for key in prob.keys():
        if(prob[key]):
            prob[key] = prob[key] / float(no_of_examples)
        else:
            prob[key] = (prob[key]+1) / float(no_of_examples+1) 
    return prob


# In[4]:


#Function for naive bayes 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    likelihoods_prob = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
        likelihoods_prob[cls] = defaultdict(list)
    class_probabilities = occurrences(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
            #the occurrences for each feature is saved in likelihoods
 
    for cls in classes:
        for j in range(0,cols):
             likelihoods_prob[cls][j] = occurrences(likelihoods[cls][j])
             #likelihoods_prob contain the probabilities per feature per class.
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods_prob[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             #else:
               #  class_probability *= 0
             results[cls] = class_probability
    print (results)
 


# In[5]:


if __name__ == "__main__":
    training,outcome = input()
    #test_sample contains the test sample
    test_sample = np.asarray((1,0,1,0))
    naive_bayes(training, outcome, test_sample)

