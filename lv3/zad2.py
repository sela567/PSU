import pandas as pd
import matplotlib.pyplot as plt

mtcars = pd.read_csv("mtcars.csv")

#1
mtcars.groupby("cyl")["mpg"].mean().plot(kind="bar")
plt.title("Prosjecna potrosnja po cilindrima")
plt.xlabel("Broj cilindara")
plt.ylabel("mpg")
plt.show()

#2
mtcars.boxplot(column="wt", by="cyl")
plt.title("Distribucija tezine po cilindrima")
plt.suptitle("")
plt.xlabel("Cilindri")
plt.ylabel("Tezina (1000 lbs)")
plt.show()

#3
mtcars.boxplot(column="mpg", by="am")
plt.title("Potrosnja: rucni vs automatski")
plt.suptitle("")
plt.xlabel("Mjenjac (0=automatik, 1=rucni)")
plt.ylabel("mpg")
plt.show()

#4  hp = snaga (x-os), qsec = ubrzanje (y-os), boja ovisi o mjenjacu
plt.scatter(mtcars["hp"], mtcars["qsec"], c=mtcars["am"])
plt.title("Snaga vs ubrzanje")
plt.xlabel("Snaga (hp)")
plt.ylabel("Ubrzanje (qsec)")
plt.show()