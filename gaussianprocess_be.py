import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Read the file Windows
#data = np.genfromtxt(r'C:\Users\atubb\Desktop\machinelearningbindingenergy\machinelearningbindingenergy\machinelearningbindingenergy\EXP2016.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))

# Read the file Ubuntu
data = np.genfromtxt(r'C:\Users\atubb\OneDrive\Työpöytä\kandi\EXP2020.dat', delimiter=',', skip_header=1, usecols=(4, 3, 7))


# Access the columns
Z = data[:, 0]  # Column 2
N = data[:, 1]  # Column 3
Binding_Energy = data[:, 2]  # Column 7

combined = np.vstack((Z, N)).T


# Create a Gaussian process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Train the model
Z_and_N = np.column_stack((Z, N))

X_train, X_val, y_train, y_val = train_test_split(Z_and_N, Binding_Energy, test_size=0.3, random_state=42)
y = Binding_Energy
model.fit(X_train, y_train)

# Use the model to predict binding energies for new data


y_new, sigma = model.predict(X_val, return_std=True)


binding_energy_diff = y_new - y_val

# Create a scatter plot with a color map
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_val[:,0],X_val[:,1], c=binding_energy_diff, cmap='viridis', s=50)
plt.colorbar(sc, label='Ennustettu sidosenergia - Mitattu sidosenergia')
plt.xlabel('Neutronien määrä N')
plt.ylabel('Protonien määrä Z')
plt.title('Ennustettu sidosenergia - Mitattu sidosenergia')
plt.show()

#final = np.stack((Z, N, y_new-Binding_Energy),axis=1)
#print(final)

#plt.scatter(final[:, 1], final[:, 0], c=final[:,2], cmap='viridis')
#plt.xlabel("Neutronien määrä N")
#plt.ylabel("Protonien määrä Z")
#cbar = plt.colorbar()
#cbar.set_label('Ennustettu sidosenergia - Mitattu sidosenergia')
#plt.show()
