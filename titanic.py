import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt



df_training = pd.read_csv('train.csv')

# Remove the 'Name' column; shouldn't be necessary.
# df_training = df_training.drop('Name', axis = 1)

#df_training[df_training['Cabin'].str[5].notna()].drop(['Name','Survived', 'Age', 'SibSp', 'Parch'],axis=1)

# Data Dictionary
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
