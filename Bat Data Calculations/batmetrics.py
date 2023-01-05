import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import stats

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
df = pd.read_csv('/workspaces/Biomechanics-Projects-2022-2023/Bat Data Calculations/Bat_Data.csv', delimiter="\t", skiprows=3, names=columns)

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

# Bat Tip Linear Velocity

df_velo = pd.DataFrame()

# We will need at linear time constant in frames per second, which is 480fps in this case
linear_time_constant = 1 / 480

# x linear velo calculation

df_velo['x_bat_top_cap_velo'] = df[['x-bat top cap']].diff() / 1000 / linear_time_constant

# y linear velo calculation - same as x with y values

df_velo['y_bat_top_cap_velo'] = df[['y-bat top cap']].diff() / 1000 / linear_time_constant

# y linear velo calculation - same as x and y with z values

df_velo['z_bat_top_cap_velo'] = df[['z-bat top cap']].diff() / 1000 / linear_time_constant

# velocity magnitude

velocity_squared = (df_velo['x_bat_top_cap_velo'] ** 2 + df_velo['y_bat_top_cap_velo'] ** 2 + df_velo['z_bat_top_cap_velo'] ** 2)
df_velo['velocity mag'] = (velocity_squared ** (1 / 2))

#Bat Tip Linear Acceleration 

df_acc = pd.DataFrame()

# x linear acceleration calculation, finding the difference between velocities

df_acc['x_bat_top_cap_acc'] = df_velo['x_bat_top_cap_velo'].diff() / linear_time_constant

# y linear acc calculation - same as x with y values

df_acc['y_bat_top_cap_acc'] = df_velo['y_bat_top_cap_velo'].diff() / linear_time_constant

# z linear acc calculation - same as x and y with z values

df_acc['z_bat_top_cap_acc'] = df_velo['z_bat_top_cap_velo'].diff() / linear_time_constant

# acceleration magnitude calculation
acceleration_squared = (df_acc['x_bat_top_cap_acc'] ** 2 + df_acc['y_bat_top_cap_acc'] ** 2 + df_acc['z_bat_top_cap_acc'] ** 2)
df_acc['acceleration mag'] = (acceleration_squared ** (1 / 2))

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

#Display euler angles
df4 = pd.DataFrame(euler, columns=['Roll','Pitch','Yaw'])

#Angular Velocity

#Euler degrees to radians
euler_rad = np.deg2rad(euler)
dfrad = pd.DataFrame(euler_rad,columns=['x','y','z'])

angular_velocity_x = dfrad[['x']].diff() / 1000 / linear_time_constant
angular_velocity_y = dfrad[['y']].diff() / 1000 / linear_time_constant
angular_velocity_z = dfrad[['z']].diff() / 1000 / linear_time_constant

df5 = pd.DataFrame()
df5 = pd.concat([angular_velocity_x, angular_velocity_y, angular_velocity_z], axis=1)
df5.loc[0] = pd.Series({'x': 0, 'y': 0, 'z': 0})

#Excluding outliers and interpolating the outliers
df6 = df5[np.abs(df5 - df5.mean()) <= (2 * df5.std())]
df6_interpolated = df6.interpolate()

#Plotting the Angular Velocity
#df6_interpolated['x'].plot(kind='line')
#df6_interpolated['y'].plot(kind='line')
#df6_interpolated['z'].plot(kind='line')
#plt.show()

#Angular Acceleration

angular_acceleration_x = df5[['x']].diff() / 1000 / linear_time_constant
angular_acceleration_y = df5[['y']].diff() / 1000 / linear_time_constant
angular_acceleration_z = df5[['z']].diff() / 1000 / linear_time_constant

df7 = pd.DataFrame()
df7 = pd.concat([angular_acceleration_x, angular_acceleration_y, angular_acceleration_z], axis=1)
df7.loc[0] = pd.Series({'x': 0, 'y': 0, 'z': 0})

#Excluding outliers and interpolating the outliers
df8 = df7[np.abs(df7 - df7.mean()) <= (2 * df7.std())]
df8_interpolated = df8.interpolate()

#Plotting the Angular Acceleration
#df8_interpolated['x'].plot(kind='line')
#df8_interpolated['y'].plot(kind='line')
#df8_interpolated['z'].plot(kind='line')
#plt.show()


# Using this to display all columns and rows
pd.option_context('display.max_rows', None, 'display.max_columns', None)
#print(df3)