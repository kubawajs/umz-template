import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

column_names = ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMetres']

report = pd.read_csv('train.tsv', sep='\t', names = column_names)

report.describe()

# report.plot()
# plt.show()

# sns.boxplot(y='price', data=report)
# plt.show()

# sns.violinplot( y='price', data=report)
# plt.show()

sns.regplot(y=report['price'], x=report['sqrMetres'])
plt.show()

reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(report, columns=['sqrMetres']), report['price'])

print(reg.predict(50)) # predict for 50 metres