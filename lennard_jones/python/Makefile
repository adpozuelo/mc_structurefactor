CUFC = nvfortran
CUFFLAGS = -shared -O3
CUFFILES = sfGpuLib.cuf
CUFSOLIB = -o sfGpuLib.so

sq : $(CUFFILES)
	$(CUFC) $(CUFFLAGS) $(CUFFILES) $(CUFSOLIB)

clean:
	rm -f *.so
	rm -f *.mod
	rm -f *.btr
	rm -f *.pyc

deepclean:
	rm -f *.so
	rm -f *.mod
	rm -f *.btr
	rm -f *.pyc
	rm -f *.dat
	rm -f *.png
	rm -f *.lammpstrj
	rm -f *.profile
	rm -f *.rdf
	rm -f *.out
	rm -f *.err
	rm -f *.ref
	rm -f log.lammps
	rm -rf __pycache__
