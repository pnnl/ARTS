ARTS Documentation
===================

**ARTS** (Abstract Runtime System) is an asynchronous many-task (AMT)
distributed runtime for data analytics, based on Open Community Runtime (OCR)
concepts.

.. rubric:: Key Features

- **Event-Driven Tasks (EDTs)** — lightweight asynchronous work units
  scheduled when all dependencies are satisfied.
- **DataBlocks (DBs)** — explicit data objects identified by globally unique
  identifiers (GUIDs); consistency is defined by the ARTS memory model
  (see :doc:`programming_model/coherence_protocols`).
- **Distributed Scheduling** — decentralised scheduler with support for
  multi-node execution over a libfabric (OFI) RDMA transport; a TCP mesh is
  retained only for process launch, the startup address exchange, and
  post-bootstrap liveness detection (no application data crosses it).
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
   programming_model/coherence_protocols
   programming_model/events
   programming_model/guids
   programming_model/finish_events

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
