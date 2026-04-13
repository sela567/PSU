import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

x = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

#podjela 80-20
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    stratify=y,
    random_state=6
)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#kreiranje i treniranje modela
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

#predikcija modela
y_pred = knn.predict(x_test)

#izracun matrice zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=['Class 0', 'Class 1']
    )
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

disp.plot(cmap=plt.cm.Blues)
plt.title('KNN - Matrica zabune')
plt.tight_layout()
plt.show()
 
#tocnost,preciznost,odziv i izvjestaj
print("Tocnost:  ", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:    ", recall_score(y_test, y_pred))
print("\nKlasifikacijski izvjestaj:")
print(classification_report(y_test, y_pred))

#e)veci broj susjeda-underfitting, malo podataka za trening
#manji broj susjeda-overfitting, puno podataka za trening
 
#f)bez skaliranja rezultati su losiji jer KNN koristi niz vrijednosti,
#a varijable (temp i CO2) imaju razlicite vrijednosti