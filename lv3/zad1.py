import pandas as pd
import numpy as np

mtcars=pd.read_csv('mtcars.csv')

#1.
print("5 automobila s najvecom potrosnjom:")
print(mtcars.sort_values(by="mpg").head(5))
print("\n")

#2.
print("3 automobila s 8 cilindara s najmanjom potrosnjom:")
auto_mpg8=mtcars[mtcars.cyl==8]
print(auto_mpg8.sort_values(by="mpg",ascending=False).head(3))
print("\n")

#3.
print("Srednja potrosnja s 6 cilindra:")
auto_mpg6=mtcars[mtcars.cyl==6]
print(auto_mpg6["mpg"].mean())
print("\n")

#4.
print("Srednja potrosnja auta s 4 cilindra i tezina izmedu 2000-2200lbs:")
auto_cyl4 = mtcars[(mtcars.cyl == 4) & (mtcars.wt >= 2.0) & (mtcars.wt <= 2.2)]
print(auto_cyl4["mpg"].mean())
print("\n")

#5.
print("Manualni mjenjac:")
auto_rucni=mtcars[mtcars.am==1]
print(len(auto_rucni))

print("Automatski mjenjac:")
auto_automatik=mtcars[mtcars.am==0]
print(len(auto_automatik))
print("\n")

#6.
print("Automatski mjesnjac i vise od 100hp")
automatik_ViseOd100=mtcars[(mtcars.am==0) & (mtcars.hp>100)]
print(len(automatik_ViseOd100))
print("\n")

#7.
print("Masa svakog auta u kg")
mtcars["masa_kg"]=mtcars.wt*0.45359237*1000
print(mtcars[["car","masa_kg"]])