from StructureFactorMonteCarlo import StructureFactorMonteCarlo


def main():
    sf = StructureFactorMonteCarlo()
    runTime = sf.loadConfigFile()
    runTime += sf.initLammps()
    runTime += sf.initVariables()
    runTime += sf.reciprocalSpace()
    runTime += sf.initStructureFactor()
    runTime += sf.runMonteCarlo()
    runTime += sf.cleanStructureFactor()

    minutes, seconds = divmod(runTime, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    days, hours = divmod(hours, 24.0)
    print(
        f"# Total wall time: {runTime:.2f}s => {int(days)}d{int(hours)}h{int(minutes)}m{int(seconds)}s"
    )


if __name__ == "__main__":
    main()
