import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Read the file
data = np.genfromtxt(r'C:\Users\atubb\Desktop\machinelearningbindingenergy\machinelearningbindingenergy\machinelearningbindingenergy\EXP2016.dat', delimiter=',', skip_header=1, usecols=(2, 3, 7))


# Access the columns
column_2 = data[:, 0]  # Column 2
column_3 = data[:, 1]  # Column 3
column_7 = data[:, 2]  # Column 7

# Print the columns
print(column_2)
print(column_3)
print(column_7)

# Create a neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000, solver='adam', alpha=0.0001, activation='relu')

# Train the model
X = np.column_stack((column_2, column_3))
y = column_7
model.fit(X, y)

# Use the model to predict binding energies for new data
new_data = np.array([[270, 161], [271, 161], [272, 162], [2, 1]]) # Example data
X_new = new_data
y_new = model.predict(X_new)



# Print predicted binding energies for new data
print(y_new)

# Draw a plot
plt.style.use('_mpl-gallery')

x = column_2
y = column_3

colors = np.random.uniform(15,80,len(x))

fig, ax = plt.subplots()

ax.scatter(x,y,s=1.5,c=colors,vmin=0,vmax=100)

plt.show()