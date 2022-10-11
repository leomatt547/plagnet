Usage
=====

.. _installation:

Installation
------------

To use PlagNet, first clone or download it as zip:

.. code-block:: console

   $ git clone https://github.com/leomatt547/plagnet.git

.. _execute:

Execute the Disposable Plastic Program
--------------------------------------

To run the program, input command as below:

.. code-block:: console

   $ python plastic_counting.py


To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

