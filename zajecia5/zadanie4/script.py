from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
import pandas as pd
from keras import optimizers


r = pd.read_csv(os.path.join("train", "train.tsv"), header=None, names=[
                "price", "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_train = pd.DataFrame(
    r, columns=["isNew", "rooms", "floor", "sqrMetres"])
Y_train = pd.DataFrame(r, columns=["price"])

def create_baseline():
    # stworzenie modelu sieci neuronowej
    model = Sequential()
    # dodanie jednego neuronu, wejście do tego neuronu to ilość cech, funkcja aktywacji sigmoid, początkowe wartości wektorów to zero.
    model.add(Dense(5, input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal'))
    model.add(Dense(3, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, activation='relu', kernel_initializer='zeros'))
    # stworzenie funkcji kosztu stochastic gradient descent
    adam_opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # kompilacja modelu
    model.compile(loss='mean_squared_error',
                  optimizer=adam_opt)

    # rysowanie architektury sieci, jeżeli ktoś ma zainstalowane odpowiednie biblioteki
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    return model


estimator = KerasRegressor(
    build_fn=create_baseline, epochs=100, verbose=True)

estimator.fit(X_train, Y_train)


# =========================================================

predictions_train = estimator.predict(X_train)

# RMSE ON TRAINING DATA:
# print('RMSE ON TRAINING DATA')
# calculate

# =========================================================

r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=[
                "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_dev = pd.DataFrame(
    r, columns=["isNew", "rooms", "floor", "sqrMetres"])

Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"),
                    header=None, names=["price"], sep='\t')

predictions_dev = estimator.predict(X_dev)

# RMSE ON DEV DATA:
# print('RMSE ON DEV DATA')
# calculate

with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction) + '\n')


# ==============================================================

r = pd.read_csv(os.path.join("test-A", "in.tsv"), header=None, names=[
                "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_test = pd.DataFrame(
    r, columns=["isNew", "rooms", "floor", "sqrMetres"])


predictions_test = estimator.predict(X_test)

# RMSE ON DEV DATA:
# print('RMSE ON DEV DATA')
# calculate

with open(os.path.join("test-A", "out.tsv"), 'w') as file:
    for prediction in predictions_test:
        file.write(str(prediction) + '\n')
