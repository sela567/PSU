from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Normalizacija i oblikovanje podataka
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

#One-hot
y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

#CNN 
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TODO: definiraj callbacks
my_callbacks = [
    callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(
    x_train_s,
    y_train_s,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks,
    verbose=1
)

#Tablica rezultata treninga
history_table = pd.DataFrame(history.history)

history_table.columns = [
    "Tocnost",
    "Gubitak",
    "Val_Tocnost",
    "Val_Gubitak"
]

print("\nTablica rezultata po epohama:\n")
print(history_table)


#TODO: Ucitaj najbolji model
best_model = keras.models.load_model('best_model.keras')

#Predikcije
y_train_pred = np.argmax(best_model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(best_model.predict(x_test_s), axis=1)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\nTocnost na skupu za ucenje:")
print(f"{train_acc * 100:.2f}%")

print("\nTocnost na skupu za testiranje:")
print(f"{test_acc * 100:.2f}%")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
cm = confusion_matrix(y_test, y_test_pred)

#Tablica matrice zabune
cm_table = pd.DataFrame(
    cm,
    index=[f"Stvarno {i}" for i in range(10)],
    columns=[f"Predvideno {i}" for i in range(10)]
)

print("\nMatrica zabune:\n")
print(cm_table)

