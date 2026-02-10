Public API (``arts.h``)
=======================

All user-facing functions are declared in ``arts.h``.  Include it as:

.. code-block:: c

   #include "arts.h"

.. contents:: Sections
   :local:
   :depth: 1


Runtime Lifecycle
-----------------

.. doxygengroup:: runtime
   :content-only:
   :members:


Memory Allocation
-----------------

.. doxygengroup:: alloc
   :content-only:
   :members:


GUID Management
---------------

.. doxygengroup:: guid
   :content-only:
   :members:


Event-Driven Tasks (EDT)
-------------------------

.. doxygengroup:: edt
   :content-only:
   :members:


Active Messages
---------------

.. doxygengroup:: active_msg
   :content-only:
   :members:


Events
------

.. doxygengroup:: event
   :content-only:
   :members:


Persistent Events
-----------------

.. doxygengroup:: persistent_event
   :content-only:
   :members:


DataBlocks (DB)
---------------

.. doxygengroup:: db
   :content-only:
   :members:


Epochs / Termination Detection
------------------------------

.. doxygengroup:: epoch
   :content-only:
   :members:


Array DataBlocks
----------------

.. doxygengroup:: arraydb
   :content-only:
   :members:


Utility Functions
-----------------

.. doxygengroup:: util
   :content-only:
   :members:
