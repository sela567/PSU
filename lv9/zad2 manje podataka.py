import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Učitavanje podataka (Cijeli skup podataka)
train_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset='training',
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

validation_ds = image_dataset_from_directory(
    directory='gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset='validation',
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

test_ds = image_dataset_from_directory(
    directory='gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

train_ds = train_ds.take(len(train_ds) // 5)
validation_ds = validation_ds.take(len(validation_ds) // 5)
test_ds = test_ds.take(len(test_ds) // 5)

model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(48, 48, 3)))

#Blok 1 - 32 filtra
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

#Blok 2-64 filtra
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

#Blok 3-128 filtara
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(43, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Callback
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

#Treniranje
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=[checkpoint]
)

#Crtanje krivulja
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Validacija')
plt.title('Točnost')
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Validacija')
plt.title('Gubitak')
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'krivulje_ucenja.png'), dpi=150)
plt.show()

best_model = keras.models.load_model(BEST_MODEL_PATH)

#Evaluacija na testnim podacima
loss, accuracy = best_model.evaluate(test_ds)
print(f'\nTočnost na testnom skupu: {accuracy * 100:.2f}%')

#matrica zabune
true_labels = []
pred_labels = []

for images, labels in test_ds:
    predictions = best_model.predict(images, verbose=0)
    true_labels.extend(np.argmax(labels, axis=1))
    pred_labels.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(true_labels, pred_labels)
print('\nMatrica zabune:')
print(cm)

#crtanje matrice zabune
plt.figure(figsize=(15, 15))
plt.imshow(cm, cmap='Blues')
plt.title('Matrica zabune')
plt.xlabel('Predviđena klasa')
plt.ylabel('Stvarna klasa')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center',
                 color='black', fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'matrica_zabune.png'), dpi=150)
plt.show()