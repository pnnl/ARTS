#!/bin/bash

cd build/examples/cpu
cp ../../../run_cxl.sh .
cp ../../../arts.cfg .
chmod +x run_cxl.sh
cp stream/stream STREAM
./run_cxl.sh ./STREAM 131072
