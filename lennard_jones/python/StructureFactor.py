import configparser as conf
import numpy as np
from lammps import lammps
import ctypes as ct
import time


class StructureFactor:
    # def __init__(self):
    #     print("----------------------")
    #     print("-- Structure Factor --")
    #     print("----------------------")

    def loadConfigFile(self, inputConfFile="sf_input.conf"):
        # print("* Loading input config file")
        time_start = time.time()
        self._config = conf.ConfigParser()
        self._config.read(inputConfFile)
        self._qMax = float(self._config["sf"]["qMax"])
        self._deltaQ = float(self._config["sf"]["deltaQ"])
        self._nIterAverage = int(self._config["sf"]["nIterAverage"])
        self._speciesChars = np.array(
            self._config["species"]["speciesChars"].split(" "), dtype=str
        )
        self._bScatteringBySpecie = np.array(
            self._config["species"]["bScatteringBySpecie"].split(" "), dtype=np.float32
        )
        self._nCudaThreads = int(self._config["cuda"]["nCudaThreads"])
        self._initLammpsFile = str(self._config["lammps"]["initLammpsFile"])
        self._ouputFile = str(self._config["output"]["ouputFile"])
        time_stop = time.time()
        # print("* Input self._config file loaded")
        return time_stop - time_start

    def initLammps(self):
        # print("* Initializing LAMMPS")
        time_start = time.time()
        self._lmp = lammps(cmdargs=["-sf", "gpu", "-nocite", "-screen", "none"])
        self._lmp.command("include " + self._initLammpsFile)
        time_stop = time.time()
        # print("* LAMMPS Initialized")
        return time_stop - time_start

    def initVariables(self):
        # print("* Initializing SF variables")
        time_start = time.time()
        self._nSpecies = len(self._bScatteringBySpecie)
        self._side = np.array(self._lmp.extract_box()[1], dtype=np.float32)
        self._nDim = len(self._side)
        self._qBoxes = int(self._qMax / self._deltaQ) + 1
        self._sQ = np.empty(self._qBoxes, dtype=np.float64)
        self._sQlen = self._qBoxes - 1
        self._deltaQVector = np.array(
            [
                self._deltaQ / 2 + (i - 1) * self._deltaQ
                for i in range(1, self._qBoxes + 1)
            ],
            dtype=np.float32,
        )
        self._lammpsExtractAtoms()
        self._tAtoms = self._lmp.numpy.extract_atom("type")
        self._nAtoms = self._lmp.get_natoms()
        self._bVector = np.zeros(self._nAtoms, dtype=np.float32)
        for i in range(self._nAtoms):
            self._bVector[i] = self._bScatteringBySpecie[self._tAtoms[i] - 1]
        self._timeFortran = np.array(0.0, dtype=np.float32)

        self._sQPtr = self._sQ.ctypes.data_as(ct.POINTER(ct.c_double))
        self._timeFortranPtr = self._timeFortran.ctypes.data_as(ct.POINTER(ct.c_float))
        self._sidePtr = self._side.ctypes.data_as(ct.POINTER(ct.c_float))
        self._bVectorPtr = self._bVector.ctypes.data_as(ct.POINTER(ct.c_float))
        self._bScatteringBySpeciePtr = self._bScatteringBySpecie.ctypes.data_as(
            ct.POINTER(ct.c_float)
        )
        time_stop = time.time()
        # print("* SF variables initialized")
        return time_stop - time_start

    def reciprocalSpace(self):
        # print("* Reciprocal space runnning")
        time_start = time.time()
        self._sfCudaLib = ct.CDLL("sfGpuLib.so").reciprocalSpace
        self._sfCudaLib.argtypes = [
            ct.POINTER(ct.c_float),  # self._timeFortranPtr (inout)
            ct.POINTER(ct.c_float),  # self._sidePtr (in)
            ct.POINTER(ct.c_float),  # self._bScatteringBySpeciePtr (in)
            ct.c_float,  # self._qMax (in)
            ct.c_float,  # self._deltaQ (in)
            ct.c_int,  # self._nCudaThreads (in)
            ct.c_int,  # self._nDim (in)
            ct.c_int,  # self._nSpecies (in)
            ct.c_int,  # self._nAtoms (in)
            ct.c_int,  # self._qBoxes (in)
        ]
        time_stop = time.time()
        time_elapsed = time_stop - time_start
        self._sfCudaLib(
            self._timeFortranPtr,
            self._sidePtr,
            self._bScatteringBySpeciePtr,
            ct.c_float(self._qMax),
            ct.c_float(self._deltaQ),
            ct.c_int(self._nCudaThreads),
            ct.c_int(self._nDim),
            ct.c_int(self._nSpecies),
            ct.c_int(self._nAtoms),
            ct.c_int(self._qBoxes),
        )
        # print("* Reciprocal space finished")
        return time_elapsed + self._timeFortran

    def initStructureFactor(self):
        # print("* Initializing Structure Factor")
        time_start = time.time()
        self._sfCudaLib = ct.CDLL("sfGpuLib.so").structureFactor
        self._sfCudaLib.argtypes = [
            ct.POINTER(ct.c_double),  # self._sQPtr (inout)
            ct.POINTER(ct.c_float),  # self._timeFortranPtr (inout)
            ct.POINTER(ct.c_float),  # self._rXYZPtr (in)
            ct.POINTER(ct.c_float),  # self._sidePtr (in)
            ct.POINTER(ct.c_float),  # self._bVectorPtr (in)
            ct.POINTER(ct.c_float),  # self._bScatteringBySpeciePtr (in)
            ct.c_int,  # iter (in)
            ct.c_int,  # self._nIterAverage (in)
            ct.c_float,  # self._qMax (in)
            ct.c_float,  # self._deltaQ (in)
            ct.c_int,  # self._nCudaThreads (in)
            ct.c_int,  # self._nDim (in)
            ct.c_int,  # self._nSpecies (in)
            ct.c_int,  # self._nAtoms (in)
            ct.c_int,  # self._qBoxes (in)
        ]
        time_stop = time.time()
        time_elapsed = time_stop - time_start
        self._sfCudaLib(
            self._sQPtr,
            self._timeFortranPtr,
            self._rXYZPtr,
            self._sidePtr,
            self._bVectorPtr,
            self._bScatteringBySpeciePtr,
            ct.c_int(0),
            ct.c_int(self._nIterAverage),
            ct.c_float(self._qMax),
            ct.c_float(self._deltaQ),
            ct.c_int(self._nCudaThreads),
            ct.c_int(self._nDim),
            ct.c_int(self._nSpecies),
            ct.c_int(self._nAtoms),
            ct.c_int(self._qBoxes),
        )
        # print("* Structure Factor initialized")
        return time_elapsed + self._timeFortran

    def runStructureFactor(self):
        # print("* Structure Factor runnning")
        time_elapsed = 0.0
        time_start = time.time()
        self._lammpsExtractAtoms()
        time_stop = time.time()
        time_elapsed += time_stop - time_start
        for iter in range(1, self._nIterAverage + 1):
            self._sfCudaLib(
                self._sQPtr,
                self._timeFortranPtr,
                self._rXYZPtr,
                self._sidePtr,
                self._bVectorPtr,
                self._bScatteringBySpeciePtr,
                ct.c_int(iter),
                ct.c_int(self._nIterAverage),
                ct.c_float(self._qMax),
                ct.c_float(self._deltaQ),
                ct.c_int(self._nCudaThreads),
                ct.c_int(self._nDim),
                ct.c_int(self._nSpecies),
                ct.c_int(self._nAtoms),
                ct.c_int(self._qBoxes),
            )
            time_elapsed += self._timeFortran
            time_start = time.time()
            self._lmp.command("run 1000")
            self._lammpsExtractAtoms()
            time_stop = time.time()
            time_elapsed += time_stop - time_start
        # print("* Structure Factor finished")
        return time_elapsed

    def cleanStructureFactor(self):
        # print("* Cleaning Structure Factor")
        self._sfCudaLib(
            self._sQPtr,
            self._timeFortranPtr,
            self._rXYZPtr,
            self._sidePtr,
            self._bVectorPtr,
            self._bScatteringBySpeciePtr,
            ct.c_int(-1),
            ct.c_int(self._nIterAverage),
            ct.c_float(self._qMax),
            ct.c_float(self._deltaQ),
            ct.c_int(self._nCudaThreads),
            ct.c_int(self._nDim),
            ct.c_int(self._nSpecies),
            ct.c_int(self._nAtoms),
            ct.c_int(self._qBoxes),
        )
        # print("* Structure Factor cleaned")
        return self._timeFortran

    def writeOutput(self):
        # print("* Writing output file")
        time_start = time.time()
        with open(self._ouputFile, "w") as file:
            for i in range(self._sQlen):
                file.write(f"{self._deltaQVector[i]} {self._sQ[i]}\n")
        time_stop = time.time()
        # print("* Output file writed")
        return time_stop - time_start

    def _lammpsExtractAtoms(self):
        self._rXYZ = self._lmp.numpy.extract_atom("x")
        self._rXYZ = np.array(self._rXYZ.T, order="F", dtype=np.float32)
        self._rXYZPtr = self._rXYZ.ctypes.data_as(ct.POINTER(ct.c_float))
