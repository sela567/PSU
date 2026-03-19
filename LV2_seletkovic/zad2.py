import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

#koristimo samo potrebne varijable
mpg=data[:,0]
cyl=data[:,1]
hp=data[:,3]
wt=data[:,5]

#b) i c)zadaci 
#(zuta-najteza,ljubicasta-najlaksa)
print('b) i c) zadataci')
plt.scatter(hp, mpg,s=wt*20,c=wt)
plt.xlabel('hp')
plt.ylabel('mpg')
plt.title('MPG vs HP')
plt.show()

#d)zadatak
print('d)zadatak')
print(f'Min mpg: {np.min(mpg)}')
print(f'Max mpg: {(np.max(mpg))}')
print(f'Srednja vrijednost MPG: {(np.mean(mpg))}')

#e)zadatak
print('e)zadatak')
mpg_6cyl=mpg[cyl==6]
print(f'Min mpg za 6 cilindra: {(np.min(mpg_6cyl))}')
print(f'Max mpg za 6 cilindra: {np.max(mpg_6cyl)}')
print(f'Srednja vrijednost mpg za 6 cilindra: {(np.mean(mpg_6cyl))}')

