import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#Read the data
df2 = pd.read_excel(r'/Users/thehoshbrown/Downloads/export (2)-2.xlsx', header= 0, usecols= [65, 71], na_values= '-')


#Change from deg to rad for plotting
df2['SpinDir'] = np.deg2rad(df2['SpinDir'])

#Fastball
FB = df2.loc[df2['type'] == ('Fastball')]
FBout = np.empty(FB['SpinDir'].shape[0], dtype=object)
FBout[:] = FB['SpinDir'].values.tolist()
FBnum= len(FBout)
FBy = [1] * FBnum

#Curveball
CB = df2.loc[df2['type'] == ('Curveball')]
CBout = np.empty(CB['SpinDir'].shape[0], dtype=object)
CBout[:] = CB['SpinDir'].values.tolist()
CBnum= len(CBout)
CBy = [1] * CBnum

#Changeup
CH = df2.loc[df2['type'] == ('Changeup')]
CHout = np.empty(CH['SpinDir'].shape[0], dtype=object)
CHout[:] = CH['SpinDir'].values.tolist()
CHnum= len(CHout)
CHy = [1] * CHnum

#Slider
SL = df2.loc[df2['type'] == ('Slider')]
SLout = np.empty(SL['SpinDir'].shape[0], dtype=object)
SLout[:] = SL['SpinDir'].values.tolist()
SLnum= len(SLout)
SLy = [1] * SLnum

#Sinker
FS = df2.loc[df2['type'] == ('Sinker')]
FSout = np.empty(FS['SpinDir'].shape[0], dtype=object)
FSout[:] = FS['SpinDir'].values.tolist()
FSnum= len(FSout)
FSy = [1] * FSnum

#Cutter
FC = df2.loc[df2['type'] == ('Cutter')]
FCout = np.empty(FC['SpinDir'].shape[0], dtype=object)
FCout[:] = FC['SpinDir'].values.tolist()
FCnum= len(FCout)
FCy = [1] * FCnum

#Splitter
SP = df2.loc[df2['type'] == ('Splitter')]
SPout = np.empty(SP['SpinDir'].shape[0], dtype=object)
SPout[:] = SP['SpinDir'].values.tolist()
SPnum= len(SPout)
SPy = [1] * SPnum

#Plot the polar graph + scatter
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection = 'polar')
ax.scatter(FBout, FBy, color='black', s = 100, label='Fastball')
ax.scatter(CBout, CBy, color='orange', s = 100, label='Curveball')
ax.scatter(CHout, CHy, color='red', s = 100, label='Changeup')
ax.scatter(SLout, SLy, color='purple', s = 100, label='Slider')
ax.scatter(FSout, FSy, color='grey', s = 100, label='Sinker')
ax.scatter(FCout, FCy, color='green', s = 100, label='Cutter')
ax.scatter(SPout, SPy, color='pink', s = 100, label='Splitter')

# Make the labels go clockwise
ax.set_theta_direction(-1)

#Place Zero at the bottom
ax.set_theta_offset(3*np.pi/2)

#Set the circumference ticks
ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))

#Set the label names
ticks = ['6', '7', '8', '9', '10', '11', '12', '1', '2', '3', '4', '5']
ax.set_xticklabels(ticks)

#Suppress the radial labels
plt.setp(ax.get_yticklabels(), visible=False)

plt.ylim(0,1.15)

plt.legend(fancybox=True, shadow=True, loc = 'center')
plt.show()