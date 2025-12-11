import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sys import platform

if platform == "win32":
    unicodeVar = "utf-16"
else:
    unicodeVar = "utf-8"

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 16})

data = np.loadtxt("sfmc_output.dat", dtype=np.float32)
epsilonRef = sigmaRef = 0.0
with open("init.lammps.nvt", "r") as file:
    for line in file:
        if line.startswith("pair_coeff"):
            line_splitted = line.split()
            epsilonRef = float(line_splitted[3])
            sigmaRef = float(line_splitted[4])

x = data[:, 0].astype(dtype=np.int32)
mse = data[:, 1]
epsilonPred = data[:, 2]
epsilonRef = np.full(epsilonPred.shape, epsilonRef, dtype=np.float32)
sigmaPred = data[:, 3]
sigmaRef = np.full(sigmaPred.shape, sigmaRef, dtype=np.float32)

fig = plt.figure(figsize=(12, 14))
axes = [plt.subplot2grid((5, 1), (row, 0), colspan=1, rowspan=1) for row in range(5)]

axes[0].plot(x, mse, "b-", label="Mean Square Error")
axes[0].set_title("Monte Carlo Structure Factor (Lennard Jones)", fontsize=16)
axes[0].legend()

axes[1].plot(x, epsilonPred, "b-", label="Epsilon Optim")
axes[1].plot(x, epsilonRef, "r-", label="Epsilon Base")
axes[1].legend()
axes[1].set_ylabel(r"$kcal/mol$")

axes[2].plot(x, sigmaPred, "b-", label="Sigma Optim")
axes[2].plot(x, sigmaRef, "r-", label="Sigma Base")
axes[2].legend()
axes[2].set_ylabel(r"$\AA$")

for ax in axes[:3]:
    ax.set_xlabel("Steps")

data = np.loadtxt("structure_factor.dat.ref", dtype=np.float32)
x = data[:, 0]
sfRef = data[:, 1]
data = np.loadtxt("structure_factor_min.dat", dtype=np.float32)
sfPred = data[:, 1]

axes[3].plot(x, sfPred, "b-", label="Structure Factor Optim")
axes[3].plot(x, sfPred, "b.")
axes[3].plot(x, sfRef, "r-", label="Structure Factor Base")
axes[3].legend()
axes[3].set_xlabel(r"$\AA^{-1}$")

os.system("tail -n 500 gdr.rdf > gdr_temp.rdf")
data = np.loadtxt("gdr_temp.rdf", dtype=np.float32)
x = data[:, 3]
gdrRef = data[:, 2]
data = np.loadtxt("gdr_min.rdf", dtype=np.float32)
gdrPred = data[:, 2]

axes[4].plot(x, gdrPred, "b-", label="RDF Optim")
axes[4].plot(x, gdrPred, "b.")
axes[4].plot(x, gdrRef, "r-", label="RDF Base")
axes[4].legend()
axes[4].set_xlabel(r"$\AA$")

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("mc_structurefactor.svg")
plt.savefig("mc_structurefactor.png")
plt.close(fig)
