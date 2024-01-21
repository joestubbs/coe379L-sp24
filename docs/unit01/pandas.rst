Introduction to Pandas 
======================

In this module, we will introduce the Python library ``pandas`` for working with the datsets 
and manipulating them [1]. After going through this module, students should be able to:

* Install and import the ``pandas`` package into a Python program.
* Understand the primary differences between the pandas ``series`` and ``dataframe``, and when to use each.
* Loading data from an external file into a pandas object. 
* Accessing pandas ``series`` and ``dataframe`` to perform various dataset manipulations.


Installing Pandas
~~~~~~~~~~~~~~~~
The ``pandas`` package is available from the Python Package Index (PyPI) and can be installed on most
platforms using a Python package mananger such as ``pip``:

.. code-block:: console

  [container/virtualenv]$ pip install pandas

Once installed, we can import the ``pandas`` package; it is customary to import the top level package 
as ``pd``, i.e., 

.. code-block:: python3
    
    >>> import pandas as pd

Now, let's take a look at the basic data structures supported by Pandas.


Pandas Series
~~~~~~~~~~~~~

A Pandas series is a one-dimensional array capable of holding data of different types 
(string, float, integer, objects, etc.) as well as axis labels. It can be thought of 
as a single column in a dataset.

We can create ``Series`` objects in different ways; for example, directly from a 
numpy array or python list: 

.. code-block:: python3
    
    >>> a = [1, 5, 8]
    >>> m = pd.Series(a)
    >>> m
    0    1
    1    5
    2    8
    dtype: int64

.. note:: 

    The Series constructor starts with a capital ``S``.

As you can see from the output every value in pandas series is labeled. If nothing 
is specified when constructing the Series, values are labeled staring from index 0 
(i.e., the first value will have index 0, the second will have index as 1, and so on).

For instance, with the previous example, we can put:

.. code-block:: python3 

    >>> m[2]
    8

However, we can customize the label indexes using the ``index`` argument 
while creating series.

.. code-block:: python3

    >>> a = [1, 5, 8]
    >>> m = pd.Series(a, index=["X", "Y","Z"])
    >>> m
    X    1
    Y    5
    Z    8
    dtype: int64

And now we can use these custom lables to index the elements of the series; for example: 

.. code-block:: python3

    >>> m["Y"]
    5

Note that if we specify custom index lables, we shouldn't use the 0-based integer indexing 
to index into our series.

What happens if you try the following: 

.. code-block:: python3

    >>> m[1]
    ?

Custom labels for indexes provide part of the power of pandas; we can use lables 
to attach meaning (or "metadata") to our data columns. 

For example, say we want to create a series of back to school supplies with their cost, 
and we have a supplies list and a cost list as follows:

.. code-block:: python3 

    >>> supplies = ['Spiral_Notebook', 'Gel_Pens', 'Sticky_Notes', 'Laptop_Bag', 'Daily_Planner']
    >>> cost_supplies_dollars = [12.81, 9.99, 5.99, 23.66,10.99]

We can use these to create a Series as follows: 

.. code-block:: python3 

    >>> supplies_cost = pd.Series(cost_supplies_dollars, index=supplies)
    >>> supplies_cost
    Spiral_Notebook    12.81
    Gel_Pens            9.99
    Sticky_Notes        5.99
    Laptop_Bag         23.66
    Daily_Planner      10.99
    dtype: float64

We see that our series is indexed by the labels we gave for the prices. We can 
now access the prices using the meaningful labels, e.g., 

.. code-block:: python3 
    >>> supplies_cost['Gel_Pens']
    9.99

We can even use these custom index labels in slices, but note that the slice is 
inclusive of both endpoints; for instance, 

.. code-block:: python3 

    >>> supplies_cost["Gel_Pens":"Daily_Planner"]
    Gel_Pens          9.99
    Sticky_Notes      5.99
    Laptop_Bag       23.66
    Daily_Planner    10.99
    dtype: float64

**In-class Exercise:** 

1. Try accessing multiple elements of the supplies_cost series at positions 1,3 and 0.

2. What will be the output of following code?

.. code-block:: python3

    >>> supplies_cost[:'Laptop_Bag']


Pandas DataFrame
~~~~~~~~~~~~~~~~

The dataframe is perhaps the most important and useful data structure in Pandas. A Pandas 
dataframe is similar to a 2d-array that can hold heterogeneous data and labeled axes. You can 
think of a dataframe as representing a spreadsheet or a database table with multiple columns. 
Said differently, a dataframe is like a dictionary of Series objects. 

Let's look at some examples to make it more clear. 

To begin, suppose we had information on employees at UT Austin. If we were storing this information 
in a spreadsheet, we might have several columns, such as: 

* Name
* EID
* Department 
* Location 

Each employee could be thought of as a row in our spreadsheet with values for each of the columns above. 
For instance, we might have data on the following employees: 

* John Doe, E0124, Austin, ITS
* Luna Lau, E0125, Houston, Student Services
* Bella Tran, E1119, Austin, Accounting 
* Raj Kumar, E2048, Dallas, Finance 

We can model these columns of data using a Pandas dataframe as follows: 

.. code-block:: python3

  >>> employees = pd.DataFrame(
      {
        'eid' :['E0124', 'E0125','E1119','E2048'],
        'name':['John Doe', 'Luna Lu', 'Bella Tran', 'Raj Kumar'],
        'location':['Austin','Houston', 'Austin', 'Dallas'],
        'department':['ITS','Student Services', 'Accounting','Finance']
      }
    )

Notice that in the above example we construct the DataFrame using a Python dictionary of lists, where 
each key in the dictionary represents a column in our dataset, and the corresponding list contains the 
values for that column. 

Indexing Columns 
^^^^^^^^^^^^^^^^^
We now have several access methods for getting at the data in our DataFrame. For example, we can access 
an individual column using the associated key:

.. code-block:: python3

  >>> employees['name']
    0      John Doe
    1       Luna Lu
    2    Bella Tran
    3     Raj Kumar
    Name: name, dtype: object

This is similar to normal Python dictionary access, but notice that the output contains indexes for the employees
(i.e., the rows) as well. 


Indexing Rows
^^^^^^^^^^^^^
We can access individual rows in the data set using the ``iloc`` function, like so:

.. code-block:: python3

  >>> employees.iloc[1]
    eid                      E0125
    name                   Luna Lu
    location               Houston
    department    Student Services
    Name: 1, dtype: object

.. note:: 

    Using ``iloc`` requires the use of brackets (``[]``), not parenthesis (``()``) as with normal function 
    invocation. 

Be aware that one *cannot* index into the DataFrame using an integer (row) index; it will result in an error:

.. code-block:: python3

  >>> employees[1]
    ---------------------------------------------------------------------------
    KeyError                                  Traceback (most recent call last)
    File ~/.cache/pypoetry/virtualenvs/risd-course-KKx7_8Y0-py3.11/lib/python3.11/site-packages/pandas/core/indexes/base.py:3791, in Index.get_loc(self, key)
    3790 try:
    -> 3791     return self._engine.get_loc(casted_key)
    3792 except KeyError as err:
    . . . 

This is the same error one would get if one tried to index a normal Python dictionary using 
an integer index (or any other index that didn't exist in the key set).


More On the ``iloc`` and ``loc`` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use ``iloc`` to select multiple rows and even specific columns for each 
row. The syntax in its general form takes two lists of integers representing the rows and 
columns we want to select, like this: 

.. code-block:: python3

    >>> df.iloc[ [<rows to select>], [<colums to select>] ]

For example: 

.. code-block:: python3

    # select rows 0, 1 and 3 and all columns
    >>> employees.iloc[[0,1,3]]
        eid 	    name 	location    department
    0 	E0124 	John Doe 	Austin 	    ITS
    1 	E0125 	Luna Lu 	Houston     Student Services
    3 	E2048 	Raj Kumar 	Dallas      Finance

And: 

.. code-block:: python3 

    # select rows 1 and 2 and columns 0, 1 and 3
    >>> employees.iloc[[1,2], [0,1,3]]
        eid 	name 	    department
    1 	E0125 	Luna Lu     Student Services
    2 	E1119 	Bella Tran  Accounting    

The ``loc`` function works similarly to ``iloc`` except that it uses integer indexes for the rows and 
string labels for the indexes instead of integers. The general format is like this: 

.. code-block:: python3 

    >>> df.loc[ [<rows (as ints>)], [<columns (as strings)>] ]

For example, 

.. code-block:: python3 

    >>> employees.loc[[0,2], ['department', 'eid']]
 	department  eid
    0 	ITS         E0124
    2 	Accounting  E1119

.. note::

    Remember, the ``i`` is for integer; always use integer indexes with ``iloc`` and 
    string label indexes with ``loc``. 

Loading Data From External Files 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will often be loading data from external files. Pandas makes it easy to create a DataFrame from 
a structured (e.g., sql file) or semi-structure (e.g., CSV) file. Here, we look at loading data from a 
CSV, but there are functions for loading data from many other sources. See the documentation on the ``io``
module for more details [2].

The basics of loading data from an external file are simple -- just use the associated function for the 
type of data you have. For CSV, that is ``pd.read_csv()``. 

DataSets on the Class Repo
^^^^^^^^^^^^^^^^^^^^^^^^^^
To show the ``read_csv()`` function, we'll download a couple of csv files from the class github repository. 
In general, the class github repository is where we will host a number of datasets for the class throughout 
the semester, including the datasets for the first three projects. 

In general, the datasets will be hosted within the ``datasets`` top-level directory, organized by unit. 
You can explore the datasets by navigating to the following URL: 

..  note:: 

    Class DataSets URL: https://github.com/joestubbs/coe379L-sp24/tree/master/datasets

Functions on DataFrames 
~~~~~~~~~~~~~~~~~~~~~~~

There are a number of important functions that we will use throughout the semester. In this 
section, we introduce a few. 





**In-Class Exercise**


References and Additional Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Pandas Documentation (2.2.0). https://pandas.pydata.org/docs/index.html
2. Input/Output: Pandas Documentation (2.2.0). https://pandas.pydata.org/docs/reference/io.html