import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from funkcija_6_1 import generate_data

X = generate_data(500, 1)

Z = linkage(X, method='ward') 

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Podaci")
plt.ylabel("Udaljenost")
plt.show()

# single -> spaja najblize tocke
# complete -> spaja najudaljenije, daje kompaktnije klastere
# ward -> minimizira klastere pri spajanju, najcesce daje najsmislenije rezultate
# average -> racunanje udaljenosti izmedu klastera, kompleksan

# Agglomerative clustering krece od n klastera i spaja ih 
# korak po korak dok ne ostane jedan -> Dendrogram to pokazuje