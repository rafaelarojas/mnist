import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

x_train_norm = x_train_norm.reshape(x_train_norm.shape[0], 28, 28, 1)
x_test_norm = x_test_norm.reshape(x_test_norm.shape[0], 28, 28, 1)

model_cnn = Sequential([
    Conv2D(32, kernel_size=(5,5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(5,5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_cnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_cnn = model_cnn.fit(x_train_norm, y_train_cat, epochs=10, validation_split=0.2)
cnn_training_time = time.time() - start_time

model_cnn.save('model_cnn.h5')

cnn_evaluation = model_cnn.evaluate(x_test_norm, y_test_cat)

print(f"CNN - Tempo de Treinamento: {cnn_training_time} segundos")
print(f"CNN - Desempenho: {cnn_evaluation}")

plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Acurácia Treino')
plt.plot(history_cnn.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Perda Treino')
plt.plot(history_cnn.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()