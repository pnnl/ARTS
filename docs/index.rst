ARTS Documentation
===================

**ARTS** (Abstract Runtime System) is an asynchronous many-task (AMT)
distributed runtime for data analytics, based on Open Community Runtime (OCR)
concepts.

.. rubric:: Key Features

- **Event-Driven Tasks (EDTs)** — lightweight asynchronous work units
  scheduled when all dependencies are satisfied.
- **DataBlocks (DBs)** — explicit data objects identified by globally unique
  identifiers (GUIDs) and managed through the CDAG memory model.
- **Distributed Scheduling** — decentralised scheduler with support for
  multi-node execution via TCP or RDMA networking.
- **GPU Support** — optional CUDA-based EDT and DataBlock operations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Programming Model

   programming_model/index
   programming_model/edt
   programming_model/datablocks
   programming_model/events
   programming_model/guids
   programming_model/epochs

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   configuration/index
   configuration/arts_cfg
   configuration/counters_cfg

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/fib

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/public_api
   api/types
   api/macros


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
