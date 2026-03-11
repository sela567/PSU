fname = input("Ime datoteke: ")

try:
    f = open(fname)
except:
    print("Datoteka se ne može otvoriti:", fname)
    quit()

broj = 0
ukupno = 0.0

for line in f:
    if line.startswith("X-DSPAM-Confidence:"):
        dijelovi = line.split(":")
        vrijednost = float(dijelovi[1])
        ukupno += vrijednost
        broj += 1

average = ukupno / broj
print("Average X-DSPAM-Confidence:", average)