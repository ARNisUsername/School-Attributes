import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


grades = pd.read_csv("school_grades_dataset.csv", sep=',')

#Change in all the letter ones from string to integer
d = {'F':0, 'M':1}
grades['sex'] = grades['sex'].map(d)
d = {'R':0, 'U':1}
grades['address'] = grades['address'].map(d)
d = {'LE3': 0, 'GT3':1}
grades['famsize'] = grades['famsize'].map(d)
d = {'A':0, 'T':1}
grades['Pstatus'] = grades['Pstatus'].map(d)

d = {'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
grades['Mjob'] = grades['Mjob'].map(d)
grades['Fjob'] = grades['Fjob'].map(d)

d = {'home':0, 'reputation':1, 'course':2, 'other':3}
grades['reason'] = grades['reason'].map(d)

d = {'mother':0, 'father':1, 'other':2}
grades['guardian'] = grades['guardian'].map(d)

d = {'no':0, 'yes':1}
for string in ('schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'):
    grades[string] = grades[string].map(d)

X = grades.drop(['G3', 'school'], axis=1)
y = grades['G3']


#Make the actual ridge data
gradeData = Ridge(alpha=100).fit(X, y)
#Get all the attributes into a list and the mean amount
allAtr = [line.rstrip('\n') for line in open('schoolAtr.txt', 'r').readlines()]
avgAtr = [int(grades[string].mean()) for string in allAtr]
#Make a library to find the location of an attribute in the list
atrLib = {}
for i in range(len(allAtr)):
    atrLib[allAtr[i]] = i
whichAtr = input("Which attribute do you want to check? ")
numChoice = int(input("How many different choices are there "))

#Makes a list with the predicted datasets for all different attributes, by changing the targeted one and making everything else average
differences = []
for i in range(numChoice):
    avgAtrStored = avgAtr
    avgAtr.pop(atrLib[whichAtr])
    avgAtr.insert(atrLib[whichAtr], i)
    differences.append(gradeData.predict(np.array([avgAtr])))
    avgAtr = avgAtrStored
#Calculates different than scores each attribute based on how close it is to the mean
avgDifferent = np.average(differences)
differentFromAvg = [float(Score - avgDifferent) for Score in differences]
for i in range(len(differentFromAvg)):
    print("{}: {}".format(i, differentFromAvg[i]))

'''
What each value means(ordered_:

What gender
What age(15-22)
Urban or rural
Family size(less than 3, greater than 3)
Parents living together or apart
Mothers education(none, primary, 5th-9th, secondary, higher)
Fathers education(none, primary, 5th-9th, secondary, higher)
Mothers job(teacher, health, services, at home, other)
Fathers job(teacher, health, services, at home, other)
Reason for school(home, reputation, course preference, other)
Guardian(mother, father, or other)
Travel time to school(less than 15 min, 15-30, 30-60, >60)
Study time( <2 hours per week, 2-5 hours, 5-10 hours, or 10+)
Number of class failures from 0-3
Extra educational support?
Extra family support
Extra paid activities
Do you do extra activities
Did you attend nursery school?
Do you want to go to higher school?
Do you have internet access at school?
In a romantic relationship?
Quality of family relationships from 1-5
Free time after school from 1-5
How much do you go out with friends? 1-5
Your parents workday alchohol consumption 1-5
Parents weekend alchohol consumption 1-5
Your curent health status 1-5
Number of school absences 0-93
First period grade 0-20
Second period grade 0-20
'''

