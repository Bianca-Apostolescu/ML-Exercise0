import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import and prepare data
data_bc = pd.read_csv('breast-cancer-diagnostic.shuf.lrn.csv')
y = data_bc[['ID', 'class']].copy()
X = data_bc.drop(['ID', 'class'], axis=1)
X_comp = pd.read_csv('breast-cancer-diagnostic.shuf.tes.csv')
y_comp = X_comp['ID'].copy()
X_comp = X_comp.drop(['ID'], axis=1)

# Prepare training and test sets using holdout method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scaling and preparation of classifier
pipe = make_pipeline(StandardScaler(), Perceptron(max_iter=10000, eta0=0.1, random_state=5))
pipe.fit(X_train, y_train['class'])
y_pred = pipe.predict(X_test)

# Output performance metrics
print('PERCEPTRON')
accuracy = accuracy_score(y_test['class'], y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test['class'], y_pred)
print("Classification Report:\n", class_report)

# Prepare training and test sets for cross-validation
scores = cross_val_score(pipe, X, y['class'], cv=5)
print(f'Cross-validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Scaling and preparation of classifier
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=17, weights='distance'))
pipe.fit(X_train, y_train['class'])
y_pred = pipe.predict(X_test)

# Output performance metrics
print('K-NN')
accuracy = accuracy_score(y_test['class'], y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test['class'], y_pred)
print("Classification Report:\n", class_report)

# Prepare training and test sets for cross-validation
scores = cross_val_score(pipe, X, y['class'], cv=5)
print(f'Cross-validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Scaling and preparation of classifier
pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion='entropy', max_depth=10))
pipe.fit(X_train, y_train['class'])
y_pred = pipe.predict(X_test)

# Output performance metrics
print('DECISION TREE')
accuracy = accuracy_score(y_test['class'], y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test['class'], y_pred)
print("Classification Report:\n", class_report)

# Prepare training and test sets for cross-validation
scores = cross_val_score(pipe, X, y['class'], cv=5)
print(f'Cross-validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Scaling and preparation of classifier
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe.fit(X_train, y_train['class'])
y_pred = pipe.predict(X_test)

# Output performance metrics
print('RANDOM FOREST')
accuracy = accuracy_score(y_test['class'], y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test['class'], y_pred)
print("Classification Report:\n", class_report)

# Prepare training and test sets for cross-validation
scores = cross_val_score(pipe, X, y['class'], cv=5)
print(f'Cross-validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Scaling and preparation of classifier
pipe = make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs', alpha=0.0001, activation='tanh', random_state=5))
pipe.fit(X_train, y_train['class'])
y_pred = pipe.predict(X_test)

# Output performance metrics
print('MULTI LAYER PERCEPTRON')
accuracy = accuracy_score(y_test['class'], y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test['class'], y_pred)
print("Classification Report:\n", class_report)

# Prepare training and test sets for cross-validation
scores = cross_val_score(pipe, X, y['class'], cv=5)
print(f'Cross-validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Predict unclassified test set
pipe.fit(X, y['class'])
y_pred = pipe.predict(X_comp)

# Write output file
out = pd.DataFrame(y_pred, y_comp)
out.rename({0: 'class'}, axis='columns', inplace=True)
out.to_csv('result.csv')
