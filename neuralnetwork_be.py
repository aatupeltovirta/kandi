import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Read the file
#data = np.genfromtxt(r'C:\Users\atubb\Desktop\machinelearningbindingenergy\machinelearningbindingenergy\machinelearningbindingenergy\EXP2020.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

data = np.genfromtxt(r'/home/aajamape/kandi/EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

# Binding_energy_calculated = a_v*A - a_s*A^(2/3) - a_c*(Z(Z-1)/A^(1/3)) - a_a*((A-2Z)^2/2)
# a_v = 15.8
# a_s = 18.3
# a_c = 0.714
# a_a = 23.2

# Access the columns
Z = data[:, 0]  # Column 2
N = data[:, 1]  # Column 3
Binding_Energy = data[:, 2]  # Column 7

combined = np.vstack((Z, N)).T

# Create a neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000, solver='adam', alpha=0.0001, activation='relu')

# Train the model
X = np.column_stack((Z, N))
y = Binding_Energy
model.fit(X, y)

# Use the model to predict binding energies for new data

predicted_be = model.predict(combined)


# Print predicted binding energies for new data
print(predicted_be)
print(Binding_Energy)


final = np.stack((Z, N, predicted_be-Binding_Energy),axis=1)
#print(final)

plt.scatter(final[:, 1], final[:, 0], c=final[:,2], cmap='viridis')
plt.ylabel("Protonien määrä Z")
plt.xlabel("Neutronien määrä N")
cbar = plt.colorbar()
cbar.set_label('Ennustettu sidosenergia - Mitattu sidosenergia')
plt.show()