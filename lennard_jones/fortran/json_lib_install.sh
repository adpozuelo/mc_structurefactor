#!/bin/bash
conda install -y -c conda-forge fpm
mkdir json_lib
git clone https://github.com/jacobwilliams/json-fortran.git
cd json-fortran/
export FPM_FC=nvfortran
fpm build --profile=release
cd ..
cp json-fortran/build/nvfortran_*/json-fortran/libjson-fortran.a json_lib/.
cp json-fortran/build/nvfortran_*/*.mod json_lib/.
rm -rf json-fortran