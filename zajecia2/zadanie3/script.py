import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

sign_dict = { 'g': 1, 'b': 0 }
rtrain = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', header=None)

notes = open('notes', 'w')


# SAVE TEST NOTES
notes.write('ZBIÓR TESTOWY\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Rozkład próby treningowej (%): ')
notes.write(str(sum(rtrain[0] == 'g') / len(rtrain)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Dokładność algorytmu zero rule: ')
notes.write(str(1 - sum(rtrain[0] == 'g') / len(rtrain)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# CREATE LR MODEL
lr = LogisticRegression()

X = pd.DataFrame(rtrain.loc[:, 1:])
lr.fit(X, rtrain[0])

# TEST PARAMETERS
TP = sum((lr.predict(X) == rtrain[0]) & (lr.predict(X) == 'g'))
TN = sum((lr.predict(X) == rtrain[0]) & (lr.predict(X) == 'b'))
FP = sum((lr.predict(X) != rtrain[0]) & (lr.predict(X) == 'g'))
FN = sum((lr.predict(X) != rtrain[0]) & (lr.predict(X) == 'b'))

notes.write('Dokładność: ')
notes.write(str((TP + TN) / len(rtrain)))
notes.write('\n')
notes.write('Czułość: ')
notes.write(str(TP / (TP + FN)))
notes.write('\n')
notes.write('Swoistość: ')
notes.write(str(TN / (FP + TN)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# MATRIX
notes.write('Macierz błędu:')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('True Positives: ')
notes.write(str(TP))
notes.write('\n')
notes.write('True Negatives: ')
notes.write(str(TN))
notes.write('\n')
notes.write('False Positives: ')
notes.write(str(FP))
notes.write('\n')
notes.write('False Negatives: ')
notes.write(str(FN))
notes.write('\n')
notes.write('-'*100)
notes.write('\n')
notes.write('\n')

###########################################

# LOAD DEV DATA
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', header=None)
rdev = pd.DataFrame(rdev)
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])

# SAVE DEV NOTES
notes.write('ZBIÓR DEV')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Rozkład próby DEV (%): ')
notes.write(str(sum(rdev_expected['y'] == 'g') / len(rdev_expected)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('Dokładność algorytmu zero rule: ')
notes.write(str(1 - sum(rdev_expected['y'] == 'g') / len(rdev)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# DEV PARAMETERS
TP = sum((lr.predict(rdev) == rdev_expected['y']) & (lr.predict(rdev) == 'g'))
TN = sum((lr.predict(rdev) == rdev_expected['y']) & (lr.predict(rdev) == 'b'))
FP = sum((lr.predict(rdev) != rdev_expected['y']) & (lr.predict(rdev) == 'g'))
FN = sum((lr.predict(rdev) != rdev_expected['y']) & (lr.predict(rdev) == 'b'))

notes.write('Dokładność: ')
notes.write(str((TP + TN) / len(rdev)))
notes.write('\n')
notes.write('Czułość: ')
notes.write(str(TP / (TP + FN)))
notes.write('\n')
notes.write('Swoistość: ')
notes.write(str(TN / (FP + TN)))
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')

# MATRIX
notes.write('Macierz błędu:')
notes.write('\n')
notes.write('-' * 100)
notes.write('\n')
notes.write('True Positives: ')
notes.write(str(TP))
notes.write('\n')
notes.write('True Negatives: ')
notes.write(str(TN))
notes.write('\n')
notes.write('False Positives: ')
notes.write(str(FP))
notes.write('\n')
notes.write('False Negatives: ')
notes.write(str(FN))
notes.write('\n')
notes.write('-'*100)

notes.close()

# PREDICT FOR DEV DATA

file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr.predict(rdev)):
   file.write(str(line)+'\n')

file.close()

# PREDICT FOR TEST DATA

rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', header=None)
rtest = pd.DataFrame(rdev)

file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr.predict(rtest)):
   file.write(str(line)+'\n')

file.close()