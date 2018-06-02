from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
import pandas as pd
from keras import optimizers


r = pd.read_csv(os.path.join("train", "train.tsv"), header=None, names=[
                "price", "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_train = pd.DataFrame(
    r, columns=["isNew", "rooms", "floor", "location", "sqrMetres"])
Y_train = pd.DataFrame(r, columns=["price"])


def create_baseline():
    # stworzenie modelu sieci neuronowej
    model = Sequential()
    # dodanie jednego neuronu, wejście do tego neuronu to ilość cech, funkcja aktywacji sigmoid, początkowe wartości wektorów to zero.
    model.add(Dense(3, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='zeros'))
    # stworzenie funkcji kosztu stochastic gradient descent
    sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # kompilacja modelu
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    # rysowanie architektury sieci, jeżeli ktoś ma zainstalowane odpowiednie biblioteki
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    return model


estimator = KerasRegressor(
    build_fn=create_baseline, epochs=50, verbose=True)


estimator.fit(X_train, Y_train)
predictions_train = estimator.predict(X_train)

# ACCURACY ON TRAINING DATA:
print('ACCURACY ON TRAINING DATA')
print((predictions_train == Y_train).mean())


r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=[
                "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_dev = pd.DataFrame(
    r, columns=["isNew", "rooms", "floor", "location", "sqrMetres"])

Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"),
                    header=None, names=["price"], sep='\t')

predictions_dev = estimator.predict(X_dev)
print('ACCURACY ON DEV DATA')
print((predictions_dev == Y_dev).mean())

with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction[0]) + '\n')
