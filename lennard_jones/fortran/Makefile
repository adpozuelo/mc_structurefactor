FC = nvfortran
FFLAGS = -O3
FINCLUDE = -I./json_lib
FFILES = /usr/local/lammps/fortran/lammps.f90 common.cuf input.cuf \
	space.cuf sf.cuf main.cuf
FEXEC = -o mcsf.exe
FLIBS = ./json_lib/libjson-fortran.a
LLIBS = -L/usr/local/lammps/fortran -L/usr/local/lammps/src -llammps

all: $(FFILES)
	$(FC) $(FFLAGS) $(FINCLUDE) $(FFILES) $(FEXEC) $(FLIBS) $(LLIBS)

clean:
	rm -f *.o
	rm -f *.a
	rm -f *.so
	rm -f *.mod
	rm -f *.btr
	rm -f *.pyc
	rm -f *.exe

deepclean:
	rm -f *.so
	rm -f *.mod
	rm -f *.btr
	rm -f *.pyc
	rm -f *.exe
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
