import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

x = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# Scatter plot
plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(x[mask, 0], x[mask, 1], label=class_names[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()

#a)
#kada je CO2 viši, prostorija je zauzetija(manje prazna)
#kada su CO2 i temperatura niži, prostorija je praznija

#b)
print("\nBroj podatkovnih primjera:", len(y)) 
#x-ulazni podaci
#y-vektor izlaznih vrijednosti, broj primjera


#c)
print("\nRazdioba podatkovnih primjera po klasama: ")
unique,counts =np.unique(y,return_counts=True)
print("Klase: ",unique)
print("Broj: ",counts)

