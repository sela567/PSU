import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


df=pd.read_csv('occupancy_processed.csv')

feature_names=['S3_Temp','S5_CO2']
target_name='Room_Occupancy_Count'

x = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

#podjela 80-20
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    stratify=y,
    random_state=6
)

#kreiranje i istrenirnje stabla odlucivanja
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

# Matrica zabune
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Class 0','Class 1']
)

disp.plot(cmap=plt.cm.Blues)
plt.title("stablo odlucivanja-matrica zabune")
plt.show()

#tocnost, preciznost, odziv i izvjestaj
print("Tocnost:  ", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:    ", recall_score(y_test, y_pred))

#izvjestaj
print(classification_report(y_test, y_pred))

#stablo odlucivanja
plt.figure(figsize=(12,8))
plot_tree(
    dt,
    filled=True
)

#vizualizacija
plt.title('Stablo odlucivanja')
plt.tight_layout()
plt.show()

#b)povećanjem max_depth stablo postaje složenije i moze doći do overfittinga
#smanjenjem max_depth model postaje jednostavniji i moze doći do underfittinga
 
#c)skaliranje ne utječe na rezultate jer se koriste pragovi a ne udaljenosti
 