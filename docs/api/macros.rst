Compiler Macros (``defs.h``)
=================================

ARTS abstracts compiler-specific attributes through portable macros
defined in ``defs.h``.  These macros work with both GCC and Clang
under strict ``-std=c17`` mode.

.. contents:: On this page
   :local:
   :depth: 1

Generic Wrapper
---------------

.. doxygendefine:: ARTS_ATTRIBUTE

Structure and Symbol Attributes
-------------------------------

.. doxygendefine:: ARTS_PACKED
.. doxygendefine:: ARTS_WEAK
.. doxygendefine:: ARTS_WEAK_IMPORT
.. doxygendefine:: ARTS_PURE

Alignment
---------

.. doxygendefine:: ARTS_ALIGNED
.. doxygendefine:: ARTS_ALIGNED_MAX

Branch Prediction
-----------------

.. doxygendefine:: ARTS_LIKELY
.. doxygendefine:: ARTS_UNLIKELY

Extended Types
--------------

.. doxygentypedef:: arts_uint128_t
