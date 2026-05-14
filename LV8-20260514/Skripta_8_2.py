import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'test.png'

# 1. Ucitaj sliku
img_original = mpimg.imread(filename)
if img_original.shape[2] == 4:
    img_original = img_original[:, :, :3]

img = color.rgb2gray(img_original)

# Prikazi sliku
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.title("Slika za klasifikaciju")
plt.axis('off')  
plt.show()

# 2. Pripremi sliku - ulaz u mrezu (28x28x1)
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# TODO: ucitaj izgradenu mrezu
# Koristimo putanju do modela koji je spremljen u 1. zadatku
try:
    model = models.load_model('models/best_model.keras') # ili 'best_model.h5' ovisno kako si spremio
    print("Model uspjesno ucitan.")
except:
    print("Greska: Model nije pronadjen. Prvo pokreni zadatak 1.")

# TODO: napravi predikciju za ucitanu sliku pomocu mreze
predikcija = model.predict(img)
klasa = np.argmax(predikcija) # Uzimamo index najveće vjerojatnosti

# TODO: ispis rezultat u terminal
print(f"\nRezultat klasifikacije: Znamenka {klasa}")
print(f"Vjerojatnost: {predikcija[0][klasa]*100:.2f}%\n")
