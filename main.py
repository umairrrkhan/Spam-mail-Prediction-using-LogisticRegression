import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data for analyzing all the data mush have in csv file 

data=pd.read_csv('mail_data.csv')

# We are seeing so many null values so we have to remove it 
mail=data.where((pd.notnull(data)),'')

mail.head()
# this is the meathod of to calculate the rows and colum
mail.shape

mail.loc[mail['Category'] ==  'spam' , 'Category',] = 0

mail.loc[mail['Category'] ==  'ham' , 'Category',] = 1

# spam =0 
# ham = 1

X=mail ['Message']

Y= mail [ 'Category']

print(X)

print(Y)

X_train, X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.2 , random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

feature_extraction =TfidfVectorizer(min_df= 1 , stop_words= 'english' , lowercase= 'True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train =Y_train.astype('int')
Y_test =Y_test.astype('int')

print(X_train)

print( X_train_features)

model= LogisticRegression()

model.fit(X_train_features , Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train , prediction_on_training_data)

print("Accuracy on training data : " , accuracy_on_training_data)

input_mail= ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]

input_data_features = feature_extraction.transform(input_mail)

prediction= model.predict(input_data_features)

print(prediction)

if prediction[0]==-1:
    print('Ham mail ')

else:   print('spam mail ')