ime_datoteke = "song.txt"

datoteka = open(ime_datoteke)

broj_rijeci = {}

for linija in datoteka:
    rijeci = linija.split() 
    for rijec in rijeci:
        broj_rijeci[rijec] = broj_rijeci.get(rijec, 0) + 1

rijeci_jednom = []

for rijec in broj_rijeci:
    if broj_rijeci[rijec] == 1:
        rijeci_jednom.append(rijec)

print("Rijeci koje se pojavljuju samo jednom:")
for rijec in rijeci_jednom:
    print(rijec)

print("Ukupno takvih rijeci:", len(rijeci_jednom))