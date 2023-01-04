import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Manually create columns for the txt file
columns = ['Index',
           'x-bat handle', 'y-bat handle', 'z-bat handle',
           'x-bat hands', 'y-bat hands', 'z-bat hands',
           'x-bat COM', 'y-bat COM', 'z-bat COM',
           'x-bat top cap', 'y-bat top cap', 'z-bat top cap',
           'x-bat bottom cap', 'y-bat bottom cap', 'z-bat bottom cap',
           'x-virtual marker end', 'y-virtual marker end', 'z-virtual marker end',
           'x-virtual marker center', 'y-virtual marker center', 'z-virtual marker center']
# Read txt file and organize columns
df = pd.read_csv('Bat Data Calculations/Bat_Data.csv', delimiter="\t", skiprows=3, names=columns)

#df.to_csv(r"Bat Data Calculations/Bat_Data.csv", index=None)

# Bat Tip Linear Velocity:

# We will need at linear time constant in frames per second, which is 480fps in this case
linear_time_constant = 1 / 480

# x linear velo calculation

x_bat_top_cap_velo = df[['x-bat top cap']].diff() / 1000 / linear_time_constant

# adding velo calculation to dataframe
df['x bat top cap velo'] = x_bat_top_cap_velo

# y linear velo calculation - same as x with y values

y_bat_top_cap_velo = df[['y-bat top cap']].diff() / 1000 / linear_time_constant
df['y bat top cap velo'] = y_bat_top_cap_velo

# y linear velo calculation - same as x and y with z values

z_bat_top_cap_velo = df[['z-bat top cap']].diff() / 1000 / linear_time_constant
df['z bat top cap velo'] = z_bat_top_cap_velo

# velocity magnitude

velocity_squared = (df['x bat top cap velo'] ** 2 + df['y bat top cap velo'] ** 2 + df['z bat top cap velo'] ** 2)
df['velocity mag'] = (velocity_squared ** (1 / 2))

# x linear acceleration calculation

# finding the difference between velocities

x_bat_top_cap_acc = df['x bat top cap velo'].diff() / linear_time_constant
df['x bat top cap acc'] = x_bat_top_cap_acc

# y linear acc calculation - same as x with y values

y_bat_top_cap_acc = df['y bat top cap velo'].diff() / linear_time_constant
df['y bat top cap acc'] = y_bat_top_cap_acc

# z linear acc calculation - same as x and y with z values

z_bat_top_cap_acc = df['z bat top cap velo'].diff() / linear_time_constant
df['z bat top cap acc'] = z_bat_top_cap_acc

# acceleration magnitude calculation
acceleration_squared = (df['x bat top cap acc'] ** 2 + df['y bat top cap acc'] ** 2 + df['z bat top cap acc'] ** 2)
df['acceleration mag'] = (acceleration_squared ** (1 / 2))

# Euler angles

df2 = pd.DataFrame()

#Vector Handle to VM bat end i

vector_bt_x = df['x-virtual marker end'] - df['x-bat handle']
vector_bt_y = df['y-virtual marker end'] - df['y-bat handle']
vector_bt_z = df['z-virtual marker end'] - df['z-bat handle']

mag_bt = ((vector_bt_x ** 2) + (vector_bt_y ** 2) + (vector_bt_z ** 2)) ** (1 / 2)

df2['ix'] = vector_bt_x / mag_bt
df2['iy'] = vector_bt_y / mag_bt
df2['iz'] = vector_bt_z / mag_bt

# Vector n or Handle to Hands

vector_n_x = df['x-bat hands'] - df['x-bat handle']
vector_n_y = df['y-bat hands'] - df['y-bat handle']
vector_n_z = df['z-bat hands'] - df['z-bat handle']

mag_n = ((vector_n_x ** 2) + (vector_n_y ** 2) + (vector_n_z ** 2)) ** (1 / 2)

df2['vector_n_x'] = vector_n_x / mag_n
df2['vector_n_y'] = vector_n_y / mag_n
df2['vector_n_z'] = vector_n_z / mag_n

# Cross product of both Vectors, |i| and |n|, to calculate k

i_array = df2[['ix', 'iy', 'iz']].to_numpy()
n_array = df2[['vector_n_x', 'vector_n_y', 'vector_n_z']].to_numpy()
k_array = np.cross(i_array, n_array)

# Vector j, using k and i

j_array = np.cross(k_array, i_array)

#Create a dataframe with i, j, and k
dfi = pd.DataFrame(i_array, columns=['i_x','i_y','i_z'])
dfj = pd.DataFrame(j_array, columns=['j_x','j_y','j_z'])
dfk = pd.DataFrame(k_array, columns=['k_x','k_y','k_z'])

df3 = pd.DataFrame()
df3 = pd.concat([dfi, dfj, dfk], axis=1)

#Convert df3 to Rotation Matrices

rotationmatrix = np.array(df3)
rmatrix = rotationmatrix.reshape((696,3,3))

#Rotation matrix to euler angles
r = R.from_matrix(rmatrix)
euler = r.as_euler('xyz', degrees=True)

#Euler degrees to radians
euler_rad = np.deg2rad(euler)



#Figuring out global axis

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(df['x-bat top cap'], df['y-bat top cap'], df['z-bat top cap'], 'green', label = 'Top Cap')
ax.plot3D(df['x-bat bottom cap'], df['y-bat bottom cap'], df['z-bat bottom cap'], 'red', label = 'Bottom Cap')
ax.plot3D(df['x-bat hands'], df['y-bat hands'], df['z-bat hands'], 'blue', label = 'Hands')
ax.plot3D(df['x-bat handle'], df['y-bat handle'], df['z-bat handle'], 'black', label = 'Handle')
ax.plot3D(df['x-virtual marker center'], df['y-virtual marker center'], df['z-virtual marker center'], 'orange', label = 'VM Center Bat')
ax.legend()
ax.set_zlim(zmin=0)
#plt.show()

# Using this to display all columns and rows
pd.option_context('display.max_rows', None, 'display.max_columns', None)
#print(df3)