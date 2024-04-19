import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import glob

directory = 'traffic_data'

files = glob.glob(directory + '/*.txt')
column_names = ['squared_id', 'timeinterval', 'internet_traffic']

dfs = []

# Iterate over the list of files
for file in files:
    # Read each file into a DataFrame
    df = pd.read_csv(file, sep='\t', usecols=[0, 1, 7], names=column_names)
    # Append the DataFrame to the list
    dfs.append(df_final)

# Concatenate all DataFrames into a single DataFrame
df_final = pd.concat(dfs, ignore_index=True)

print("testing github")
"""
df = pd.read_csv('sms-call-internet-mi-2013-11-02.txt', sep='\t', usecols=[0, 1, 7],names=column_names)
df['internet_traffic'] = df['internet_traffic'].replace(np.NaN,0)
#print(df.head())
summed_df = df.groupby('squared_id')['internet_traffic'].sum().reset_index()
sns.kdeplot(summed_df['internet_traffic'], shade=True)
plt.xlabel('Total Traffic Observed Over 3 Days')
plt.ylabel('Probability')
plt.title('PDF of Total Traffic by squared_id')
plt.show()
"""
