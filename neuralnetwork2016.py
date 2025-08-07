import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import statistics
import matplotlib.pyplot as plt

# Read the file
data = np.genfromtxt(r'C:\Users\atubb\kandi\EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7, 2, 9))

#data = np.genfromtxt(r'/home/aajamape/kandi/EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

# Access the columns
#Protons
Z = data[:, 0]  # Column 2
#Neutrons
N = data[:, 1]  # Column 3
#Binding Energy
Binding_Energy = data[:, 2]  # Column 7

A = data[:, 3] # Column 1

BperAA = data[:, 4]



#Binding energy / mass number
BperA = Binding_Energy / A

# Create a neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000, solver='adam', alpha=0.0001, activation='relu')

# Train the model
Z_and_N = np.column_stack((Z, N))

X_train, X_val, y_train, y_val = train_test_split(Z_and_N, BperAA, test_size=0.3, random_state=42)

y = Binding_Energy
model.fit(X_train, y_train)

# Use the model to predict binding energies for new data

predicted_be = model.predict(X_val)

final_binding_energy = y_val


#Mean squared error

mse = mean_squared_error(final_binding_energy, predicted_be)


#Calculate difference of new and known binding energies
binding_energy_diff = predicted_be - final_binding_energy

#Calculate the average
average = statistics.mean(binding_energy_diff)
max_abs_diff = np.max(np.abs(binding_energy_diff))

# Create a scatter plot with a color map
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_val[:,1],X_val[:,0], c=binding_energy_diff, cmap='seismic',vmin=-max_abs_diff, vmax=max_abs_diff, s=50)
plt.colorbar(sc, label='Ennustettu sidosenergia - Mitattu sidosenergia (MeV)')
plt.xlabel('Neutronien määrä N')
plt.ylabel('Protonien määrä Z')
plt.show()

print('Ennustuksien ja oikeiden arvojen erotuksen keskiarvo: ' +  str(average) + ' Keskineliövirhe: ' + str(mse))