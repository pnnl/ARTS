#!/bin/bash

cd build/examples/cpu
cp ../../../run_cxl.sh .
cp ../../../arts.cfg .
chmod +x run_cxl.sh
cp random_access/random_access RANDOM_ACCESS
./run_cxl.sh ./RANDOM_ACCESS
