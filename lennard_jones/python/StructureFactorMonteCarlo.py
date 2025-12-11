from StructureFactor import StructureFactor
from random import random
from math import exp
import configparser as conf
import numpy as np
from lammps import lammps
import ctypes as ct
import time
import os


class StructureFactorMonteCarlo(StructureFactor):
    # def __init__(self):
    #     print("----------------------------------")
    #     print("-- Structure Factor Monte Carlo --")
    #     print("----------------------------------")

    def loadConfigFile(self, inputConfFile="sfmc_input.conf"):
        # print("* Loading input config file")
        time_super = super().loadConfigFile(inputConfFile)
        time_start = time.time()
        self._maxIter = int(self._config["mc"]["maxIter"])
        self._msePrecission = float(self._config["mc"]["msePrecission"])
        self._kt = float(self._config["mc"]["kt"])
        self._deltaEpsilon = float(self._config["mc"]["deltaEpsilon"])
        self._minEpsilon = float(self._config["mc"]["minEpsilon"])
        self._maxEpsilon = float(self._config["mc"]["maxEpsilon"])
        self._midrangeEpsilon = (self._maxEpsilon + self._minEpsilon) / 2.0
        self._deltaSigma = float(self._config["mc"]["deltaSigma"])
        self._minSigma = float(self._config["mc"]["minSigma"])
        self._maxSigma = float(self._config["mc"]["maxSigma"])
        self._midrangeSigma = (self._maxSigma + self._minSigma) / 2.0
        self._gdrRecords = int(self._config["mc"]["gdrRecords"])
        self._gdrInputFile = str(self._config["mc"]["gdrInputFile"])
        self._gdrOutputFile = str(self._config["mc"]["gdrOutputFile"])
        self._sfRefFile = str(self._config["mc"]["sfRefFile"])
        self._outputMseFile = str(self._config["output"]["outputMseFile"])
        time_stop = time.time()
        # print("* Input config file loaded")
        return time_stop - time_start + time_super

    def initVariables(self):
        # print("* Initializing SF variables")
        time_super = super().initVariables()
        time_start = time.time()
        self._sQref = np.loadtxt(self._sfRefFile, dtype=np.float64)
        self._sQref = self._sQref[:, 1]
        self._epsilon = self._sigma = self._rc = self._rcDiv2 = self._rcDiv4 = 0.0
        with open(self._initLammpsFile, "r") as file:
            for line in file:
                if line.startswith("pair_coeff"):
                    line_splitted = line.split()
                    self._epsilon = float(line_splitted[3])
                    self._sigma = float(line_splitted[4])
                    self._rc = float(line_splitted[5])

        time_stop = time.time()
        # print("* SF variables initialized")
        return time_stop - time_start + time_super

    def runMonteCarlo(self):
        # print("* Monte Carlo runnning")
        time_super = self.runStructureFactor()
        time_start = time.time()
        mseMin = self._mse = np.sqrt(
            np.sum(np.square(np.subtract(self._sQref, self._sQ[: self._sQlen])))
        )
        nAccept = mcIter = 0
        with open(self._outputMseFile, "w", buffering=1) as file:
            file.write(f"# nAcc MeanSquareError Epsilon Sigma\n")
            file.write(
                f"{nAccept:6d} {self._mse:.9e} {self._epsilon:.5f} {self._sigma:.3f}\n"
            )
            print(f"# Step rAcc MeanSquareError Epsilon Sigma < B")
            print(
                (
                    f"{mcIter:6d} {0.0:.2f} {self._mse:.9e} {self._epsilon:.5f} "
                    f"{self._sigma:.3f} {0:d} {0:d}"
                )
            )
            while mcIter < self._maxIter:
                mcIter += 1
                epsilonTmp = self._epsilon + self._deltaEpsilon * (2 * random() - 1)
                if epsilonTmp < self._minEpsilon or epsilonTmp > self._maxEpsilon:
                    epsilonTmp = self._midrangeEpsilon
                sigmaTmp = self._sigma + self._deltaSigma * (2 * random() - 1)
                if sigmaTmp < self._minSigma or sigmaTmp > self._maxSigma:
                    sigmaTmp = self._midrangeSigma
                self._lmp.command(f"pair_coeff 1 1 {epsilonTmp} {sigmaTmp} {self._rc}")
                self._lmp.command("run 50000")
                time_stop = time.time()
                time_super += time_stop - time_start
                time_super += self.runStructureFactor()
                time_start = time.time()
                mseTmp = np.sqrt(
                    np.sum(np.square(np.subtract(self._sQref, self._sQ[: self._sQlen])))
                )
                bolzPass = 0
                msePass = mseTmp < self._mse
                if msePass:
                    nAccept += 1
                    self._epsilon = epsilonTmp
                    self._sigma = sigmaTmp
                    self._mse = mseTmp
                else:
                    diff_mse_kt = (mseTmp - self._mse) / self._kt
                    bolzPass = exp(-diff_mse_kt) > random()
                    if bolzPass:
                        nAccept += 1
                        self._epsilon = epsilonTmp
                        self._sigma = sigmaTmp
                        self._mse = mseTmp
                if self._mse < mseMin:
                    mseMin = self._mse
                    self.writeOutput(mcIter)
                    os.system(
                        "tail -n "
                        + str(self._gdrRecords)
                        + " "
                        + self._gdrInputFile
                        + " > "
                        + self._gdrOutputFile
                    )
                rAccept = nAccept / mcIter
                if msePass or bolzPass:
                    file.write(
                        f"{nAccept:6d} {self._mse:.9e} {self._epsilon:.5f} {self._sigma:.3f}\n"
                    )
                print(
                    (
                        f"{mcIter:6d} {rAccept:.2f} {self._mse:.9e} {self._epsilon:.5f} "
                        f"{self._sigma:.3f} {msePass:d} {bolzPass:d}"
                    )
                )
                if self._mse < self._msePrecission:
                    print(f"MeanSquareError precission's limit reached!")
                    break
        time_stop = time.time()
        # print("* Monte Carlo finished")
        return time_stop - time_start + time_super

    def writeOutput(self, step):
        # print("* Writing output file")
        time_start = time.time()
        with open(self._ouputFile, "w") as file:
            file.write(
                f"# step={step} mse={self._mse} epsilon={self._epsilon} sigma={self._sigma} rc={self._rc}\n"
            )
            for i in range(self._sQlen):
                file.write(f"{self._deltaQVector[i]} {self._sQ[i]}\n")
        time_stop = time.time()
        # print("* Output file writed")
        return time_stop - time_start
