#!/bin/bash

cd build/examples/cpu
cp ../../../run_cxl.sh .
cp ../../../arts.cfg .
chmod +x run_cxl.sh
./run_cxl.sh ./fibDB "$1"
