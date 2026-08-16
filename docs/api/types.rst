Core Types & Hints (``arts.h``)
===============================

This page documents the supporting types, creation hints, and internal
structures used throughout the ARTS runtime — the definitions in
``arts.h`` that are not part of the function-level groups covered by
:doc:`public_api`, plus the internal structures declared in
``runtime_types.h`` and the coherence module's ``types.h`` headers.

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


Hint Type
---------

.. doxygengroup:: hint_type
   :content-only:
   :members:


Creation Hints
--------------

Three independent hint structs (one each for EDT, DB, and event
creation) so each creation API can grow features that do not apply to
the others.

.. doxygengroup:: hint_structs
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


Event Creation Hint
-------------------

.. doxygengroup:: event_hint
   :content-only:
   :members:


User Callbacks
--------------

Optional weak-symbol callbacks (``main_edt``, ``init_per_node``,
``init_per_worker``) an application defines to hook into the runtime
lifecycle.

.. doxygengroup:: user_callbacks
   :content-only:
   :members:


Internal Structures
-------------------

These structures are managed by the runtime internals and are included
here for reference only.

.. doxygengroup:: internal_structs
   :content-only:
   :members:


Internal DB / Coherence Structures
----------------------------------

Per-DB home directory metadata and per-rank cache state used by the
coherence protocols; also managed entirely by the runtime.

.. doxygengroup:: internal_db_structs
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
