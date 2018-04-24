import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=[
                     "Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

notes = open('notes', 'w')


# SAVE TEST NOTES
notes.write('ZBIÓR TESTOWY\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Rozkład próby treningowej (%): ')
notes.write(str(sum(rtrain.Occupancy) / len(rtrain)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Dokładność algorytmu zero rule: ')
notes.write(str(1 - sum(rtrain.Occupancy) / len(rtrain)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# CREATE LR MODEL
lr = LogisticRegression()
lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy)

notes.write('Dokładność: ')
notes.write(str(sum(lr.predict(rtrain.CO2.values.reshape(-1, 1)) == rtrain.Occupancy) / len(rtrain)))
notes.write('\n')
notes.write('Czułość: ', )
notes.write('\n')
notes.write('Swoistość: ', )
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# MATRIX
notes.write('Macierz błędu:')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('True Positives: ')
notes.write(str(sum((lr.predict(rtrain.CO2.values.reshape(-1, 1)) == rtrain.Occupancy) & (lr.predict(rtrain.CO2.values.reshape(-1, 1)) == 1))))
notes.write('\n')
notes.write('True Negatives: ')
notes.write(str(sum((lr.predict(rtrain.CO2.values.reshape(-1, 1)) == rtrain.Occupancy) & (lr.predict(rtrain.CO2.values.reshape(-1, 1)) == 0))))
notes.write('\n')
notes.write('False Positives: ')
notes.write(str(sum((lr.predict(rtrain.CO2.values.reshape(-1, 1)) != rtrain.Occupancy) & (lr.predict(rtrain.CO2.values.reshape(-1, 1)) == 1))))
notes.write('\n')
notes.write('False Negatives: ')
notes.write(str(sum((lr.predict(rtrain.CO2.values.reshape(-1, 1)) != rtrain.Occupancy) & (lr.predict(rtrain.CO2.values.reshape(-1, 1)) == 0))))
notes.write('\n')
notes.write('-'*100)
notes.write('\n')
notes.write('\n')

###########################################

# LOAD DEV DATA
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])

# SAVE DEV NOTES
notes.write('ZBIÓR DEV')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Rozkład próby DEV (%): ')
notes.write(str(sum(rdev_expected['y']) / len(rdev_expected)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Dokładność algorytmu zero rule: ')
notes.write(str(1 - sum(rdev_expected['y']) / len(rdev)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

notes.write('Dokładność:')
notes.write(str(sum(lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) / len(rdev)))
notes.write('\n')
notes.write('Czułość:', )
notes.write('\n')
notes.write('Swoistość:', )
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# MATRIX
notes.write('Macierz błędu:')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('True Positives: ')
notes.write(str(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 1))))
notes.write('\n')
notes.write('True Negatives: ')
notes.write(str(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0))))
notes.write('\n')
notes.write('False Positives: ')
notes.write(str(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 1))))
notes.write('\n')
notes.write('False Negatives: ')
notes.write(str(sum((lr.predict(rdev.CO2.values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev.CO2.values.reshape(-1, 1)) == 0))))
notes.write('\n')
notes.write('-'*100)

notes.close()

# PREDICT FOR DEV DATA

file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr.predict(rdev.CO2.values.reshape(-1, 1))):
   file.write(str(line)+'\n')

file.close()

# PREDICT FOR TEST DATA

rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rtest = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr.predict(rtest.CO2.values.reshape(-1, 1))):
   file.write(str(line)+'\n')

file.close()

# SAVE PLOT

sns.regplot(x=rdev.CO2, y=rdev_expected.y, logistic=True, y_jitter=.1)
plt.savefig("wykres")