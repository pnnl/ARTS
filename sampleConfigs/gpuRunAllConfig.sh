#! /bin/bash

if [ $# -eq 0 ]; then
  echo "$0 APP [APP_ARGS]"
  exit 1
fi

app=$1
app_args=${@:2}

for cfgFile in `ls *.cfg`
do
    export artsConfig=$cfgFile
    echo "Running $app $app_args with $cfgFile"
    $app $app_args
done