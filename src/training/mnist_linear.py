import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model_mlp = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model_mlp.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_mlp = model_mlp.fit(x_train_norm, y_train_cat, epochs=20, validation_split=0.2)
mlp_training_time = time.time() - start_time

model_mlp.save('model_linear.h5')

mlp_evaluation = model_mlp.evaluate(x_test_norm, y_test_cat)

print(f"MLP - Tempo de Treinamento: {mlp_training_time} segundos")
print(f"MLP - Acurácia: {mlp_evaluation[1]}")
