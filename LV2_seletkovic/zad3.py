import numpy as np
import matplotlib.pyplot as plt

img=plt.imread('tiger.png')
img=img[:, :,0].copy()
img_a=img+0.9

img_a=np.clip(img_a, 0, 1) 
plt.figure()
plt.imshow(img_a, cmap="gray")
plt.title('a) Svjetlija slika')
plt.show()

#b)zarotirati sliku
img_b=np.rot90(img,k=-1) 
plt.figure()
plt.imshow(img_b, cmap="gray")
plt.title('b) Rotirana slika')
plt.show()

#c)zrcaljena slika
img_c=np.fliplr(img) 
plt.figure()
plt.imshow(img_c, cmap="gray")
plt.title('c) Zrcaljena slika')
plt.show()

#d)smanjena rezolucija
img_d=img[::10,::10]
plt.figure()
plt.imshow(img_d, cmap="gray")
plt.title('d) Smanjena rezolucija')
plt.show()