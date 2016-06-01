# Titanic

# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:01:13 2016

@author: AISJFASSIER
"""
import pylab as P
import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

							# Then convert from a list to an array.
train_df = pd.read_csv('C:/Jeanne/Python/train.csv', header=0) 
test_df = pd.read_csv('C:/Jeanne/Python/test.csv', header=0)        # Load the train file into a dataframe
test_df['Survived']=0
frame=[train_df,test_df]
combi_df = pd.concat(frame)
important = ['Survived']
reordered = important + [c for c in combi_df.columns if c not in important]
combi_df = combi_df[reordered]

combi_df['Gender'] = combi_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
if len(combi_df.Embarked[ combi_df.Embarked.isnull() ]) > 0:
    combi_df.Embarked[ combi_df.Embarked.isnull() ] = combi_df.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(combi_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
combi_df.Embarked = combi_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
mean_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        mean_ages[i,j] = combi_df[(combi_df['Gender'] == i) & \
                              (combi_df['Pclass'] == j+1)]['Age'].dropna().mean() 
combi_df['AgeFill'] = combi_df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        combi_df.loc[ (combi_df.Age.isnull()) & (combi_df.Gender == i) & (combi_df.Pclass == j+1),\
                'AgeFill'] = mean_ages[i,j]
combi_df['AgeIsNull'] = pd.isnull(combi_df.Age).astype(int)
combi_df['Child'] = combi_df['Age']<15
combi_df['Child2'] = combi_df['Child'].map( {False: 0, True: 1} ).astype(int)
combi_df['CabinIsNull'] = pd.isnull(combi_df.Cabin).astype(int)
combi_df['Deck'] = combi_df['Cabin'].str[:1]
if len(combi_df.Deck[ combi_df.Deck.isnull() ]) > 0:
    combi_df.Deck[ combi_df.Deck.isnull() ] = 'Z'
Decks = list(enumerate(np.unique(combi_df['Deck'])))    # determine all values of Embarked,
combi_df['Decks'] = combi_df['Deck'].map( {'A': 0, 'B': 1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7,'Z':8} ).astype(int)
combi_df['FamilySize'] = combi_df['SibSp'] + combi_df['Parch']
combi_df['Age*Class'] = combi_df.AgeFill * combi_df.Pclass
combi_df['Age*Class'].hist()
P.show()
name1 = pd.DataFrame(combi_df.Name.str.split(', ',1).tolist(),columns = ['left','right'])
name2 = pd.DataFrame(name1.right.str.split('.',1).tolist(),columns = ['left','right'])
combi_df['Title']=name2['left']
title_counts = combi_df.groupby('Title').size()
print("\n"+'Title Counts')
print(title_counts)
title_map = {'Capt': 1, 
             'Col': 2, 'Major': 2,
             'Don': 3, 'Sir': 3,
             'Dr': 4, 
             'Dona':5, 'Jonkheer': 5, 'Lady': 5, 'the Countess':5,            
             'Master': 6,
             'Miss': 7, 'Mlle': 7, 'Ms': 7,
             'Mme': 8, 'Mrs': 8,
             'Mr': 9,
             'Rev': 10}
title_map2 = {'Capt': 1, 
             'Col': 1, 'Major': 1,
             'Don': 1, 'Sir': 1,
             'Dr': 1, 
             'Dona':2, 'Jonkheer': 2, 'Lady': 2, 'the Countess':2,            
             'Master': 2,
             'Miss': 2, 'Mlle': 2, 'Ms': 2,
             'Mme': 2, 'Mrs': 2,
             'Mr': 1,
             'Rev': 1}
combi_df['TitleInt'] = combi_df['Title'].map(title_map).astype(int)
combi_df['TitleInt2'] = combi_df['Title'].map(title_map2).astype(int)
mean_fares_test = np.zeros((1,3))
for j in range(0, 3):
    mean_fares_test[0,j] = combi_df[(combi_df['Pclass'] == j+1)]['Fare'].dropna().mean()
mean_fares_test
for j in range(0, 3):
    combi_df.loc[ (combi_df.Fare.isnull()) & (combi_df.Pclass == j+1),\
                'Fare'] = mean_fares_test[0,j]




combi_df.dtypes
combi_df = combi_df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Deck','Title','TitleInt2','Age','Child','PassengerId'], axis=1) 
combi_df = combi_df.dropna()
train_df=combi_df[:891]
test_df=combi_df[891:]
test_df = test_df.drop(['Survived'], axis=1) 

combi_df.describe() 



###################################################################################
############ PREDICTION

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("C:/Jeanne/Python/forest JF4.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'




combi_df2= combi_df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Deck','Title','Age','Child','PassengerId'], axis=1) 
combi_df2 = combi_df2.dropna()
train_df2=combi_df[:891]
test_df2=combi_df[891:]
test_df2 = test_df2.drop(['Survived'], axis=1) 

combi_df.describe() 



###################################################################################
############ PREDICTION 2

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("C:/Jeanne/Python/forest JF3.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
