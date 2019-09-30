#! /bin/bash

if [ $# -eq 0 ]; then
  echo "$0 NUM_GPU CFG_FILE APP [APP_ARGS]"
  exit 1
fi

#function join { local IFS="$1"; shift; echo "$*"; }

num_gpus=$1
cfg=$2
app=$3
app_args=${@:4}

if [ -z $LAUNCHER ]; then
  echo "LAUNCHER not set. So using SSH"
  echo "export LAUNCHER, SRUN_HOST to use slurm"
fi

if [ "$LAUNCHER" == "slurm" ]; then
  launcher="srun"
  if [ -z $HOST ]; then
    echo "export SRUN_HOST=\e[3mHOST_TO_LAUNCH_SRUN\e[0m"
    exit 1
  fi
  host=$SRUN_HOST
fi

if [ $num_gpus == 0 ]; then
  echo "$0 NUM_GPU APP [APP_ARGS]"
  echo "Please provide NUM_GPU > 0"
  exit 1
fi

echo "Running $app $app_args on $num_gpus GPUs with $cfg"

for (( i=1; i<$((num_gpus+1)); ++i ))
do
  echo "Now running on $i gpus"
  sed -i -e "s/gpu\s=\s.*/gpu = $i/" $2
  export artsConfig=$2
  if [ -z $launcher ]; then
    $app $app_args
  else
    $launcher -p $SRUN_HOST $app $app_args
  fi
done
