import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import graphviz


df_training = pd.read_csv('train.csv')

# The following removes some seemingly-unnecessary columns. (Possible
# reason why this is unwise: some family members with same surname are
# in data.)

df_training = df_training.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1)


# The following shows the percent of survivors based on their ticket class
# df_training.loc[:,['Pclass','Survived']].groupby('Pclass').mean()

# The following shows the percent of survivors and other stats of
# passengers based on their sex

# df_training.loc[:,['Sex','Survived']].groupby('Sex').mean()

# The following shows the percent of survivors and other stats of
# passengers based on their sex and ticket class
# df_training.loc[:,['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean()

# After this preliminary analysis of the data, it seems a decision tree is in order. Here goes:

#Sklearn requires numerical data, so first the Sex is converted to 1 for male 0 for female:
df_training_numerical=pd.get_dummies(df_training, drop_first= True)

# The data has missing (i.e. 'NaN') AGE's, which sklearn does not
# allow. Filling the NaN values with -9999, +9999, 0, 33, 50 resulted
# in 80% accuracy, even with entropy vs. gini, and different MAX_DEPTH
# settings (Those settings can be adjusted below).
df_training_numerical['Age']=df_training_numerical['Age'].fillna(-9999)

# Instead we try binning the as follows. (Still getting around 80% accuracy.)

df_training_numerical['Age']=pd.cut(df_training_numerical['Age'],bins=[-10000,-9999,17,54,155], labels=[-1,0,1,2])

X = df_training_numerical.drop(['Survived'],axis=1)
y = df_training_numerical[['Survived']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


from sklearn import tree
dec_tree_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
dec_tree_clf = dec_tree_clf.fit(X_train,y_train)

train_score = dec_tree_clf.score(X_train,y_train)
test_score = dec_tree_clf.score(X_test,y_test)



# The following draws the tree as "titanic_decision.pdf"
dot_data = tree.export_graphviz(dec_tree_clf, out_file= None,
                                feature_names=list(X.columns)) 
graph = graphviz.Source(dot_data)
graph.render("titanic_decision")

# time to test our model
y_pred = dec_tree_clf.predict(X_test)

print("Accuracy:", np.round(sklearn.metrics.accuracy_score(y_test, y_pred),3))












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
