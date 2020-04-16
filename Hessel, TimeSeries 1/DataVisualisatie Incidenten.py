import pandas as pd
import numpy as np
import seaborn as sns
import pickle 

import matplotlib as mpl
import matplotlib.pyplot as figuur
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Wijzig de intstellingen voor de te tonen plots

figuur.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
plt = figuur

# seaborn
sns.set(style="whitegrid", color_codes=True)

# Laad gegevens uit aangemaakte pickle in de DataManipulatie Incidenten.py
df = pickle.load(open('TSeries','rb'))

df['open'] = df.open.astype('float32')
print(df)
tijdr = df
tijdr = tijdr.set_index('maand')
# Resamplen alleen nodig als er ontbrekende waarden zijn in de tijdreeks.
# dat is hier niet het geval
# tijdr = tijdr.open.resample('M').mean()
print(tijdr)
# print(df.head())

plt.figure(figsize=(16, 5), dpi=120)
plt.gca().set(title="", xlabel='jjjjmm', ylabel='Incidenten')
plt.plot(df['datum'], df['open'], 'r', label='Alle')
# plt.plot(df['maand'], df['Storing'], 'b', label='Storing')
# plt.plot(df['maand'], df['Overig'], 'g', label='Overig')
plt.legend(loc='upper left')


# Prepareren van de data
df['datum2'] = pd.to_datetime(df.datum, format='%Y-%m-%d %H:%M:%S')
df['jaar'] = [d.year for d in df.datum2]
df['mnd'] = [d.strftime('%b') for d in df.datum2]
jaren = df['jaar'].unique()

# Selecteren kleuren
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(jaren), replace=False)

# Tekenen van de plot met linegraph
slplt = figuur
slplt.figure(figsize=(16,12), dpi=100)
for i, j in enumerate(jaren):
    if i > 0:
        slplt.plot('mnd', 'open', data=df.loc[df.jaar == j, :], color=mycolors[i], label=j)
        slplt.text(df.loc[df.jaar == j, :].shape[0]-0.9, df.loc[df.jaar==j, 'open'][-1:].values[0], j, fontsize=12, color=mycolors[i])

# Tekenen van de jaarlijkse trend/seizoensinvloeden
sbplt = figuur
fig, axes = sbplt.subplots(1, 2, figsize=(20, 7), dpi=80)
sns.boxplot(x='jaar', y='open', data=df, ax= axes[0])
sns.boxplot(x='mnd', y='open', data=df.loc[~df.jaar.isin([2013, 2019]),:])

# Zet de titels
axes[0].set_title('Trend (boxplot)', fontsize=18)
axes[1].set_title('Seizoen (boxplot)', fontsize=18)
