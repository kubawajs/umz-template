import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model

column_names = ['price', 'mileage', 'year', 'brand', 'engineType', 'engineCapacity']

# load data
report = pd.read_csv('../train/in.tsv', sep='\t', names = column_names)
report.describe()

# convert strings to int values
brand_dict = {}

for id, brand in enumerate(report.brand.unique()):
    brand_dict[brand] = id

# replace brands with int values
report = report.replace({"brand": brand_dict})

# create linear regression model
reg = linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['mileage', 'year', 'engineCapacity', 'brand']), report['price'])


## predict prices for dev data
in_file_col_names = column_names[1:]
in_file = pd.read_csv('in.tsv', sep='\t', names = in_file_col_names)

# replace brands with int values
in_file['brand'] = in_file.brand.map(brand_dict).fillna(-1).astype(int)

result = reg.predict(pd.DataFrame(in_file, columns=['mileage', 'year', 'engineCapacity', 'brand']))
result = pd.Series(np.maximum(result, 0)) # remove results below 0
result.to_csv('out.tsv', sep='\t', header=False, index=False)


## predict prices for test data
in_file_col_names = column_names[1:]
in_file = pd.read_csv('../test-A/in.tsv', sep='\t', names = in_file_col_names)

# replace brands and engine types with int values
in_file['brand'] = in_file.brand.map(brand_dict).fillna(-1).astype(int)

result = reg.predict(pd.DataFrame(in_file, columns=['mileage', 'year', 'engineCapacity', 'brand']))
result = pd.Series(np.maximum(result, 0)) # remove results below 0
result.to_csv('../test-A/out.tsv', sep='\t', header=False, index=False)