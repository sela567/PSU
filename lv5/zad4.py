import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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

log_reg = LogisticRegression(random_state=42)

#treniranje podataka
log_reg.fit(x_train, y_train)

#predikcija
y_pred = log_reg.predict(x_test)

#matrica zabune
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Class 0','Class 1']
)

disp.plot(cmap=plt.cm.Blues)
plt.title("Logisticka regresija - matrica zabune")
plt.show()

#tocnost
print("Tocnost: ", accuracy_score(y_test, y_pred))

#report
print(classification_report(y_test, y_pred))

#logicka regresija nam daje lošije rezultate od KNN-a i stabla odlucivanja
#razlog slabijih rezultata je ta jer je granica odlucivanja linearna a podaci nisu