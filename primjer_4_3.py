import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cars_processed.csv")

print(df.info())
#broj auta
print("\nBroj auta:", len(df))

#tipovi stupca
print(df.dtypes)

#cijene
print("\nNajskuplji:\n", df.loc[df.selling_price.idxmax()])
print("\nNajjeftiniji:\n", df.loc[df.selling_price.idxmin()])

#auta proizvedena 2012
print("\nAuta proizvedena 2012: ", len(df[df['year']==2012]))

#max/min km
print("\nNajviše km:\n", df.loc[df.km_driven.idxmax()])
print("\nNajmanje km:\n", df.loc[df.km_driven.idxmin()])

#najcesci broj sjedala
print("\nNajčešći broj sjedala:", df.seats.mode()[0])
print("\nProsječna km (Diesel):", round(df[df.fuel=="Diesel"].km_driven.mean(), 2))
print("\nProsječna km (Petrol):", round(df[df.fuel=="Petrol"].km_driven.mean(), 2))

#prosjecna kilometraza dizel auta
print("dizel:",round(df[df["fuel"]=="Diesel"]["km_driven"].mean(),2))
print("benzin:",round(df[df["fuel"]=="Petrol"]["km_driven"].mean(),2))

#vizualno
sns.pairplot(df, hue="fuel")
sns.scatterplot(data=df, x="km_driven", y="selling_price", hue="fuel")

df = df.drop(columns=["name", "mileage"])

#grafovi
df.boxplot(column="selling_price", by="fuel")
df["selling_price"].hist()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.tight_layout()
plt.show()