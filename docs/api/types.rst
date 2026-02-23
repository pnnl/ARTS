Core Types (``rt.h``)
=====================

This page documents the core types and internal structures defined in
``rt.h``.  These types are used throughout the ARTS runtime.

.. contents:: On this page
   :local:
   :depth: 1

Fundamental Types
-----------------

.. doxygengroup:: core_types
   :content-only:
   :members:


Type / Access-Mode Enumeration
------------------------------

.. doxygengroup:: type_enum
   :content-only:
   :members:


Dependency Types
----------------

.. doxygengroup:: dep_types
   :content-only:
   :members:


Event Slot Types
----------------

.. doxygengroup:: event_slots
   :content-only:
   :members:


GUID Range and Array DB
-----------------------

.. doxygengroup:: range_array
   :content-only:
   :members:


Termination Detection
---------------------

.. doxygengroup:: td_types
   :content-only:
   :members:


Buffer
------

.. doxygengroup:: buffer_type
   :content-only:
   :members:


Internal Structures
-------------------

These structures are managed by the runtime internals and are included
here for reference only.

.. doxygengroup:: internal_structs
   :content-only:
   :members:


GUID Layout (``guid.h``)
------------------------

The GUID is a 64-bit integer with three fields accessed via shift/mask
macros (``ARTS_GUID_GET_TYPE``, ``ARTS_GUID_GET_RANK``, ``ARTS_GUID_GET_KEY``,
``ARTS_GUID_MAKE``).  See :doc:`/programming_model/guids` for the bit layout.


Utility
-------

.. doxygenfunction:: arts_printf
