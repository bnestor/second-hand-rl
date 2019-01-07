"""
plot_results.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats


df=pd.read_csv('output3.csv', header=None)
df[1]=[1 if item in ['[1]', 'True'] else 0 for item in df[1]]


df=df.apply(pd.to_numeric, errors = 'coerce')
print(df.head(5))

figure=plt.figure()

a2c=plt.scatter(df[df[1]==1].index, df.loc[df[1]==1, 3], alpha=0.6, label='a2c')
controller=plt.scatter(df[df[1]==0].index, df.loc[df[1]==0, 3], alpha=0.6, label='controller')
plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
plt.ylabel('Time to complete task')

plt.legend(handles=[a2c, controller])

plt.show()


fig=plt.figure()

x=np.linspace(0, 100)
blend_x=1-scipy.stats.norm(0, 5).cdf(x)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

plt.fill_between(x, 0, blend_x, facecolor='green')
plt.fill_between(x, 1, blend_x, facecolor='magenta')

plt.xlabel('Distance from magenta point')
plt.ylabel('Fraction of error contributed by magenta')


plt.show()