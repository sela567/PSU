import numpy as np
import matplotlib.pyplot as plt 

def sahovnica(kvadrat,visina,sirina):
    bijeli=np.ones((kvadrat,kvadrat))*255 
    crni=np.zeros((kvadrat,kvadrat))  

    redovi=[]
    
    for i in range(visina): 
        red=[]
        
        for j in range(sirina):
            if(i+j)%2==1:
                red.append(bijeli) 
            else:
                red.append(crni) 
        redovi.append(np.hstack(red))
        
    return np.vstack(redovi)

img=sahovnica(50,4,5)
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.show()
