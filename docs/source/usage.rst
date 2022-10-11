Usage
=====

.. _installation:

Installation
------------

To use PlagNet, first clone or download it as zip:

.. code-block:: console

   (.venv) $ git clone https://github.com/leomatt547/plagnet.git

Execute the Disposable Plastic Program
--------------------------------------

To run the program, input command as below:

.. code-block:: console

   $ python plastic_detection.py

Plastic Exchange Configuration 
------------------------------
For several objects that have detected as disposable plastic bag, the program will count how many reusable plastic bag that you can get.
The configuration for exchange condition can be seen in as follow:

.. code-block:: console
   
   # Set plastic bag exchange conditions here
   small_cond = 15  # items
   medium_cond = 10  # items
   large_cond = 5  # items

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

