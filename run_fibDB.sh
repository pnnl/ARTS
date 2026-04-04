#!/bin/bash

cd build/examples/cpu
cp ../../../run_cxl.sh .
cp ../../../arts.cfg .
chmod +x run_cxl.sh
for i in {1..100}; do
  ./run_cxl.sh ./fibDB 25 &> fibDB_out_${i}.log
done
cd ../../../
python3 check_logs_fibDB.py