import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data_bc = pd.read_csv('breast-cancer-diagnostic.shuf.lrn.csv')
id_class = data_bc[['ID', 'class']].copy()
data_bc = data_bc.drop(['ID', 'class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_bc, id_class, test_size=0.2, random_state=5)
y_train = y_train.drop('ID', axis=1)
y_test_ids = y_test['ID'].copy()
y_test = y_test.drop('ID', axis=1)
pipe = make_pipeline(StandardScaler(), Perceptron(max_iter=100, eta0=0.1, random_state=5))
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

out = pd.DataFrame(y_pred, y_test_ids)
out.rename({0: 'class'}, axis='columns', inplace=True)
out.to_csv('result.csv')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
