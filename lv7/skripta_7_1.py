import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model=keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(784,)))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train, y_train_s, epochs=4, batch_size=32)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test_s, verbose=0)

print("Tocnost na train skupu:", train_acc)
print("Tocnost na test skupu:", test_acc)

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(cm)


# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
greska = np.where(y_pred != y_test)[0]

plt.figure(figsize=(10, 6))
for i in range(10):
    idx = greska[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title("Tocno:" + str(y_test[idx]) + " \nPredikcija:" + str(y_pred[idx]))
    plt.axis("off")
plt.show()


