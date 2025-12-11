from StructureFactor import StructureFactor


def main():
    sf = StructureFactor()
    runTime = sf.loadConfigFile()
    runTime += sf.initLammps()
    runTime += sf.initVariables()
    runTime += sf.reciprocalSpace()
    runTime += sf.initStructureFactor()
    runTime += sf.runStructureFactor()
    runTime += sf.cleanStructureFactor()
    runTime += sf.writeOutput()

    minutes, seconds = divmod(runTime, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    days, hours = divmod(hours, 24.0)
    print(
        f"# Total wall time: {runTime:.2f}s => {int(days)}d{int(hours)}h{int(minutes)}m{int(seconds)}s"
    )


if __name__ == "__main__":
    main()
