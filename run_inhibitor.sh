#!/bin/bash

cd build/examples/cpu
cp ../../../run_cxl.sh .
cp ../../../arts.cfg .
chmod +x run_cxl.sh
echo "Running Sequential Access"
./run_cxl.sh ./inhibitor_raw_mem -n $(( $1 * $2 )) -p s -s 8096 -w 0 -t 0
echo "Running Linear Access"
./run_cxl.sh ./inhibitor_raw_mem -n $(( $1 * $2 )) -p l -s 8096 -w 0 -t 1024
echo "Running Random Access"
./run_cxl.sh ./inhibitor_raw_mem -n $(( $1 * $2 )) -p r -s 8096 -w 0 -t 0
