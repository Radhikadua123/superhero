
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os


# In[3]:



train_dir = os.listdir("CAX_Superhero_Train/")
train_len =len(train_dir)
print(train_len)

test_dir = os.listdir("CAX_Superhero_Test/")
test_len =len(test_dir)
print(test_len)


# In[4]:


train_data =[]; train_label =[]; map1=[]
for i in range(train_len):
    temp =[]
    temp.append(i)
    temp.append(train_dir[i])
    map1.append(temp)
    if(train_dir[i] != "superhero.ipynb" and train_dir[i] != ".ipynb_checkpoints" ):
        
        print(train_dir[i],i)
        
        train_dir_class1 = os.listdir("CAX_Superhero_Train/"+train_dir[i])
        train_class1_len =len(train_dir_class1)
        print(train_class1_len)
           
        for j in range(train_class1_len):
            #print(os.path.exists("/home/isha/superhero/CAX_Superhero_Train/"+train_dir[i]+"/"+train_dir_class1[j]))
            img = cv2.imread("CAX_Superhero_Train/"+train_dir[i]+"/"+train_dir_class1[j])
            train_data.append(img)
            train_label.append(i)
            #cv2.imshow('image',img)


# In[5]:


print(map1)
train_data = np.array(train_data)
train_label = np.array(train_label)
print(train_data.shape, train_label.shape)


# In[7]:


train_data_resized = []
for i in range(len(train_data)):
    train_data_resized.append(cv2.resize(train_data[i], (32,32)))


# In[8]:


train_data_resized = np.array(train_data_resized)
print(train_data_resized.shape)


# In[9]:


s = train_data_resized.shape

train_data_final = np.reshape(train_data_resized,(s[0], s[1]* s[2]* s[3]))
print(train_data_final.shape)



# In[10]:


### 

training_data = train_data_final
training_label = train_label


# In[11]:


test_data =[] ; test_label =[]
for i  in range(test_len):
    img = cv2.imread("CAX_Superhero_Test/"+test_dir[i])
    test_data.append(img)


# In[19]:



test_data = np.array(test_data)
print(test_data.shape)


# In[20]:


test_data_resized = []
for i in range(len(test_data)):
    test_data_resized.append(cv2.resize(test_data[i], (32,32)))


# In[21]:


test_data_resized = np.array(test_data_resized)
print(test_data_resized.shape)


# In[22]:


s_test = test_data_resized.shape

test_data_final = np.reshape(test_data_resized,(s_test[0], s_test[1]* s_test[2]* s_test[3]))
print(test_data_final.shape)


# In[31]:


from sklearn.model_selection import train_test_split
X_train1, X_val1, y_train1, y_val1 = train_test_split(training_data, training_label, test_size=0.33, shuffle = True)
print(X_train1.shape, y_val1.shape)


# In[32]:


X_test1 = test_data_final


# In[33]:


from sklearn import svm


# In[37]:


clf = svm.SVC()
clf.fit(X_train1[:,:], y_train1[:])


# In[38]:


from sklearn.metrics import accuracy_score

val_preds = clf.predict(X_val1)
print(" TRaining Accuracy %0.2f" % (100*accuracy_score(val_preds, y_val1)))


# In[ ]:


test_preds = clf.predict(X_val1)


# In[ ]:


clf = MLPClassifier(solver='adam', alpha=1e-5hidden_layer_sizes=(128,))

