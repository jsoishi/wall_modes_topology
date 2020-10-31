import h5py
import matplotlib.pyplot as plt
import sys

filename = sys.argv[-1]

df = h5py.File(filename, 'r')
t = df['scales/sim_time'][:]

TE = df['tasks/Energy_x'][:,0,0,0] + df['tasks/Energy_y'][:,0,0,0] + df['tasks/Energy_z'][:,0,0,0]
TEns = df['tasks/Enstrophy_x'][:,0,0,0] + df['tasks/Enstrophy_y'][:,0,0,0] + df['tasks/Enstrophy_z'][:,0,0,0]

plt.figure(figsize=(8,16))
plt.subplot(3,1,1)
plt.plot(t, df['tasks/Energy_x'][:,0,0,0], label='x')
plt.plot(t, df['tasks/Energy_y'][:,0,0,0], label='y')
plt.plot(t, df['tasks/Energy_z'][:,0,0,0], label='z')
plt.plot(t, TE, color='k', linewidth=2, label='total')
plt.legend()
plt.xlabel("time")
plt.ylabel("Energy")

plt.subplot(3,1,2)
plt.plot(t, df['tasks/Nusselt'][:,0,0,0])
plt.xlabel("time")
plt.ylabel("Nusselt")

plt.subplot(3,1,3)
plt.plot(t, df['tasks/Enstrophy_x'][:,0,0,0], label='x')
plt.plot(t, df['tasks/Enstrophy_y'][:,0,0,0], label='y')
plt.plot(t, df['tasks/Enstrophy_z'][:,0,0,0], label='z')
plt.plot(t, TEns, color='k', linewidth=2, label='total')
plt.legend()
plt.xlabel("time")
plt.ylabel("Enstrophy")

plt.tight_layout()
plt.savefig("energies.png", dpi=300)

