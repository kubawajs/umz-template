import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

column_names = ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMetres']

report = pd.read_csv('../train/train.tsv', sep='\t', names = column_names)

report.describe()

sns.regplot(y=report['price'], x=report['sqrMetres'])
img = plt.gcf()
img.savefig('plot.png')

reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(report, columns=['sqrMetres']), report['price'])

in_file_col_names = column_names[1:]
in_file = pd.read_csv('in.tsv', sep='\t', names = in_file_col_names)

result = reg.predict(pd.DataFrame(in_file['sqrMetres']))
result = pd.Series(result)
result.to_csv('out.tsv', sep='\t', header=False, index=False)