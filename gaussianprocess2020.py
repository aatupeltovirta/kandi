import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF
import statistics
import matplotlib.pyplot as plt

# Read the file Windows
data = np.genfromtxt(r'C:\Users\atubb\kandi\EXP2020.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7,9))

# Read the file Ubuntu
#data = np.genfromtxt(r'/home/aajamape/kandi/EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))


# Access the columns
Z = data[:, 0]  # Column 2
N = data[:, 1]  # Column 3
Binding_Energy = data[:, 2]  # Column 7
BperA = data[:, 3]

# Create a Gaussian process model
kernel = RBF(1, (1, 3))
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Train the model
Z_and_N = np.column_stack((Z, N))

X_train, X_val, y_train, y_val = train_test_split(Z_and_N, BperA, test_size=0.3, random_state=42)

model.fit(X_train, y_train)

# Use the model to predict binding energies for new data

y_new = model.predict(X_val)

#Mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val, y_new)

#Calculate difference of new and known binding energies
binding_energy_diff = y_new - y_val

#Keskiarvon laskeminen
average = statistics.mean(binding_energy_diff)

# Create a scatter plot with a color map
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_val[:,1],X_val[:,0], c=binding_energy_diff, cmap='seismic', s=50)
plt.colorbar(sc, label='Ennustettu sidosenergia - Mitattu sidosenergia')
plt.xlabel('Neutronien määrä N')
plt.ylabel('Protonien määrä Z')
plt.title('Gaussin prosessin ennustus 2020 datasta')
plt.show()

print('Ennustuksien ja oikeiden arvojen erotuksen keskiarvo: ' +  str(average) + ' Keskineliövirhe: ' + str(mse))