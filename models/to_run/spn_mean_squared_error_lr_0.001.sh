#!/usr/bin/env bash
#PBS -k oe
#PBS -j oe

# export PATH="/usr/local/cuda-8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/caffe/build/tools"
# export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
# export PYTHONPATH="/home/caffe/python:/home/caffe2/build"
# export CAFFE_HOME="/home/caffe"
# export OMP_NUM_THREADS=16
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"

export PATH="/usr/local/cuda-9.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/doleinik/caffe/build/tools"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
export PYTHONPATH="/home/doleinik/caffe/python"
export CAFFE_HOME="/home/doleinik/caffe"
export OMP_NUM_THREADS=16
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

# echo $PBS_O_WORKDIR
# echo $PBS_O_HOME
# export PBS_O_HOME=$PBS_O_WORKDIR
# export HOME=$PBS_O_WORKDIR
# echo $PBS_O_HOME

python $PBS_O_WORKDIR/../SP_model.py spn_mean_squared_error_lr_0.001
