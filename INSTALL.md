Table of Contents
=================

*   [Project Overview](#project-overview)
    *   [Detailed Summary](#detailed-summary)
*   [Installation Guide](#installation-guide)
    *   [Environment Requirements](#environment-requirements)
    *   [Dependencies](#dependencies)
    *   [Distribution Files](#distrubution-files)
    *   [Installation Instructions](#installation-instructions)
    *   [Test Cases](#test-cases)
*   [User Guide](#user-guide)
*   [Contributors](#contributors)

Project Overview
================

**Project Name:** Abstract RunTime System (ARTS)

**Principle Investigator:** Joshua Suetterlein (Joshua.Suetterlein@pnnl.gov)

**General Area or Topic of Investigation:** Asynchronous Many Task runtime (AMT)

**Release Number:** 1.2

Detailed Summary
----------------

The ARTS runtime system is an AMT that explores macro-dataflow execution for data analytics.  This runtime provides users
with a distributed global adress space, a distributed memory model, and efficent synchronization constructs to write
efficent applications on a massively parallel system.

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software.

Environment Requirements
------------------------

**Programming Language:** C, C++, and CUDA

**Operating System & Version:** Ubuntu 16.04.3, CentOS 7, 

**Required Disk Space:** 160MB

**Required Memory:** At least 1GB

Dependencies
------------

| Name | Version | Download Location | Country of Origin | Optional/Required | Special Instructions |
| ---- | ------- | ----------------- | ----------------- | ----------------- | -------------------- |
| cmake | 3.8 | https://github.com/Kitware/CMake | USA | Required | Must use 3.8 or above for CUDA language support | 
| CUDA | 9.2.148 | https://developer.nvidia.com/cuda-92-download-archive | USA | Required | Tested with CUDA 9.2. Please check OS CUDA combination |
| CUBLAS | 9.0 | https://developer.nvidia.com/cuda-90-download-archive | USA | Optional | Typically ships with CUDA or CUDA Toolkit. |
| Thrust | 9.0 | https://developer.nvidia.com/cuda-90-download-archive | USA | Optional | Typically ships with CUDA or CUDA Toolkit. |
| HWLoc | 1.11 | https://www.open-mpi.org/software/hwloc/v1.11/ | USA | Optional | New versions not yet supported | 
Distribution Files
------------------

Key Directories
core/ - Main directory containing the runtime source files.  
example/ - Directory containing both CPU and GPU examples.  
graph/ - Directory containing graph data structures/methods for applications.  Not part of the core runtime development.  
sampleConfigs/ - Directory with arts.conf files required to run examples.  
test/ - Directory containing tests to debug issues.  For development purposes.  

Key Files:
arts.h - Include file required by arts programs.  
arts.cfg - Arts configuration file.  This file is required in the same directory as a running ARTS program.  
libarts.so - Runtime library generated after building.  Required for linking programs.  


Installation Instructions
-------------------------

Before attempting to build ARTS, please take a look at the requirements in dependencies.  While cmake will attempt to find the libraries in your path, you can help cmake by providing the path of a library using a flag -D<LIB_NAME>_ROOT=<PATH_TO_LIB_DIR> (e.g. -DHWLOC_ROOT=/usr/lib64/ or -DCUDA_ROOT=/usr/lib64/cuda9.2.148).

For CPU build only:
```
git clone <url-to-ARTS-repo>  # or untar the ARTS source code.
cd arts
mkdir build && cd build
cmake ..
make -j
```

For GPU builds:
```
git clone <url-to-ARTS-repo>  # or untar the ARTS source code.
cd arts
mkdir build && cd build
cmake .. -DCUDA_ROOT=$CUDAROOT
make -j
```

Test Cases
----------

To test CPU execution, first go to the examples directory in your build directory.  
Next cd into the cpu folder, and set the launcher (job scheduler) in the configuration file (arts.cfg).
To run an arts program the configuration file must be in the run directory.
The launcher may be set to ssh, slurm, or lsf.  By default is is set to slurm.
Next run fib:
```
./fib 10
[0] Fib 10: 55 time: 75412 nodes: 1 workers: 16
```

To test GPU execution, go to the gpu directory under the examples in your build directory.
Again set the launcher to ssh, slurm, or lsf in the configuration file.
To run the GPU code, the gpu flag in the configuration should be set to the number of disired GPUs.
Next run fibGpu
```
./fibGpu 10
[0] Fib 10: 55 time: 442941730 nodes: 1 workers: 16
[0] Cleaned 6372 bytes
[0] Occupancy :
[0] 	GPU[0] = 0.007812
[0] 	GPU[1] = 0.007812
[0] 	GPU[2] = 0.007812
[0] 	GPU[3] = 0.007812
[0] 	GPU[4] = 0.007812
[0] 	GPU[5] = 0.007812
[0] 	GPU[6] = 0.007812
[0] 	GPU[7] = 0.007812
[0] HITS: 9 MISSES: 255 FREED BYTES: 1016 BYTES FREED ON EXIT 4
[0] HIT RATIO: 0.034091

```

An example srun command for slurm users is as follows:
```
srun -N 8 -n 8 -c 20 ./fib 10
```
The example above uses 8 nodes each with 20 threads.

User Guide
==========

Troubleshooting:  

1. Make sure you are running from a directory with an arts.cfg
2. Check the launcher in arts.cfg
3. For GPU support please pull from the gpu branch

Some configurations:  
The configuration file (arts.cfg) has many options which are set to reasonable defaults.  Three options will be changed by the user, launcher, threads, and gpus.  The launcher has already been discussed.  For threads and gpus, these should be set to a the number of resources you want to use per node.
  
Please refer to arts.h and artsRT.h for documentation.  
  
To run our GPU matrix multiply please go to build/examples/mm.  Set your configuration file, and run:
```
./mmTile [matrix size] [tile size]
```
Or for the cuBLAS version:
```
./mmTileBlas [matrix size] [tile size]
```

Contributors
============

### MAIN TEAM MEMBERS

1. Joshua Suetterlein, joshua.suetterlein@pnnl.gov
2. Joseph Manzano, joseph.manzano@pnnl.gov
3. Andres Marquez, andres.marquez@pnnl.gov

### CONTRIBUTORS

1. Vinay Amatya
2. Kiran Ranganath
3. Marcin Zalewski
4. Jesun Firoz
5. Vito Castellana
6. Marco Minutoli
7. Antonino Tumeo
8. John Feo
9. Andrew Lumsdaine