#!/bin/bash
#SBATCH -J test
#SBATCH -p bcores48,bcores48-2,bcores48-3
#SBATCH -n 1
#SBATCH --error=%J.err   
#SBATCH --output=%J.out
ulimit -s unlimited

# - - - - - - - - - - USER SETTINGS - - - - - - - - - - - - #
 
# LOCATION TO RUN WPS/WRF
OUT_DIR="/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_1km"

# LOCATION OF WPS EXECUTABLES
WPS_DIR="/data/home/maguolin/WRF/Build_WRF/WPS/WPS-4.6.0"
# LOCATION OF WHERE TO DOWNLOAD REANALYSIS DATA
ICBC_DIR="/data/home/maguolin/ask/wake/rans/hurui/A/meso/meso_3km/ICBC"
# LOCATION OF WRF EXECUTABLES
EXE_DIR="/data/home/maguolin/WRF/WRFV4.6.0/run"

# SETUP COMPUTING ENVIRONMENT
#module load intel/2017u5
#export NETCDF=/data/software/wps/needed
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/wrf/deps-icc/lib

module load gcc/9.3.0
export DIR=/data/home/maguolin/WRF/Build_WRF/LIBRARIES
export PATH=$DIR/netcdf/bin:$PATH
export NETCDF=$DIR/netcdf
export PATH=$DIR/mpich/bin:$PATH
export LDFLAGS=-L$DIR/grib2/lib
export CPPFLAGS=-I$DIR/grib2/include
export HDF5=$DIR/hdf5
export LD_LIBRARY_PATH=$HDF5/lib:$LD_LIBRARY_PATH
export JASPERLIB=$DIR/grib2/lib
export JASPERINC=$DIR/grib2/include     

# - - - - - - - - - END USER SETTINGS - - - - - - - - - - - #
#===========================================================#
# If directory doesn't exist, create it
if [ ! -d $ICBC_DIR ]; then
    mkdir -p $ICBC_DIR
fi
#======================================================================
#======================================================================
#                               Run WPS               
#======================================================================
#======================================================================

ln -sf $WPS_DIR/geogrid.exe .
ln -sf $WPS_DIR/ungrib.exe .
ln -sf $WPS_DIR/metgrid.exe .
if [ ! -d geogrid ]; then
    mkdir geogrid
fi
if [ ! -d metgrid ]; then
    mkdir metgrid
fi
ln -sf $WPS_DIR/geogrid/GEOGRID.TBL geogrid/.
ln -sf $WPS_DIR/metgrid/METGRID.TBL metgrid/.

cp $WPS_DIR/link_grib.csh .
ln -sf $WPS_DIR/ungrib/Variable_Tables/Vtable.ECMWF Vtable

csh ./link_grib.csh $ICBC_DIR/*.grib .


 ./geogrid.exe
./ungrib.exe
./metgrid.exe

#======================================================================
#======================================================================
#                               Run WRF               
#======================================================================
#======================================================================
ln -sf $EXE_DIR/[aBbCcEGgHikLmopRStUV]* .
ln -sf $EXE_DIR/real.exe .
ln -sf $EXE_DIR/wrf.exe .


dirs_to_create=( auxout rsl wrfout towers wrfrst )
for dir in "${dirs_to_create[@]}" ; do
    if [ ! -d $dir ]; then mkdir $dir
    fi
done
