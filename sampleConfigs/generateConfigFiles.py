#!/usr/bin/env python3

import configparser
import itertools as it
import sys

if(len(sys.argv) == 1):
    print(sys.argv[0], "NUM_GPU LAUNCHER")
    exit(0)

if(len(sys.argv) == 3):
    launcher = sys.argv[2]
else:
    launcher = "ssh"

r2 = [str(x) for x in range(2)]
r3 = [str(x) for x in range(3)]

options = {}
options["freeDbAfterGpuRun"] = r2
options["runGpuGcIdle"] = r2
options["runGpuGcPreEdt"] = r2
options["deleteZerosGpuGc"] = r2
options["gpuFit"] = r3
options["gpuLocality"] = r3
options["gpuP2P"] = r2

config = configparser.ConfigParser()
config.optionxform=str
cfgTemplate = "arts.cfg"
config.read(cfgTemplate)

conf = config[config.sections()[0]]

conf["scheduler"] = "3"
conf["launcher"] = launcher
conf["gpu"] = sys.argv[1]

keys = sorted(options)

combinations = list(it.product(*(options[key] for key in keys)))

i = 1

for comb in combinations:
    cfgFile = "test"+str(i)+".cfg"
    for key, ith in zip(keys, list(range(len(comb)))):
        conf[key] = comb[ith]
    with open(cfgFile, 'w') as cFile:
        config.write(cFile)
    i += 1