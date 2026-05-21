import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Učitavanje modela
model = tf.keras.models.load_model('best_model.keras')

img_raw = tf.io.read_file('znak.png')
img = tf.image.decode_jpeg(img_raw, channels=3)
img = tf.image.resize(img, [48, 48])

img_array = img.numpy()
img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0

#Predikcija
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
vjerojatnost = np.max(prediction) * 100

print(f'Predviđena klasa: {predicted_class}')
print(f'Sigurnost modela: {vjerojatnost:.2f}%')

#Prikaz slike
plt.imshow(img.numpy().astype('uint8'))
plt.title(f'Klasa: {predicted_class} ({vjerojatnost:.1f}%)')
plt.show()