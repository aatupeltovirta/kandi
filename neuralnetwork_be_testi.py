import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Read the file
data = np.genfromtxt(r'C:\Users\atubb\OneDrive\Työpöytä\kandi\EXP2020.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

#data = np.genfromtxt(r'/home/aajamape/kandi/EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

# Access the columns
#Protons
Z = data[:, 0]  # Column 2
#Neutrons
N = data[:, 1]  # Column 3
#Binding Energy
Binding_Energy = data[:, 2]  # Column 7

# Create a neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000, solver='adam', alpha=0.0001, activation='relu')

# Train the model
Z_and_N = np.column_stack((Z, N))

X_train, X_val, y_train, y_val = train_test_split(Z_and_N, Binding_Energy, test_size=0.3, random_state=42)

y = Binding_Energy
model.fit(X_train, y_train)

# Use the model to predict binding energies for new data

predicted_be = model.predict(X_val)

#Mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val, predicted_be)
print("MSE on validation set:", mse)

#Calculate difference of new and known binding energies
binding_energy_diff = predicted_be - y_val

# Create a scatter plot with a color map
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_val[:,0],X_val[:,1], c=binding_energy_diff, cmap='viridis', s=50)
plt.colorbar(sc, label='Ennustettu sidosenergia - Mitattu sidosenergia')
plt.xlabel('Neutronien määrä N')
plt.ylabel('Protonien määrä Z')
plt.title('Ennustettu sidosenergia - Mitattu sidosenergia')
plt.show()
