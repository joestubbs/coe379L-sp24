Data Standardization And Pipelines
==================================

In this module, we motivate and introduce additional data pre-processing techniques that involve 
transforming the input dataset before fitting our model. We also introduce the ``Pipeline`` 
class from sklearn for chaining multiple transformations together. 

By the end of this module, students should be able to: 

1. Understand data pre-processing techniques that tranform the dataset such as standardization,
   maxabs Scaler and robust Scaler.
2. Understand when to use each technique. 
3. Implement the sklearn ``Pipeline`` abstraction to conduct multi-step data analysis.
   
Data Pre-Processing: Motivation 
--------------------------------

Most of the data analysis methods from Unit 1 assumed that the variables were on a similar numeric 
scale to one another. 
Moreover, the algorithms for fitting the models we have looked at mostly work by minimizing a cost function 
using an algorithm like gradient descent. The partial derivatives appearing in the gradient 
computation are sensitive to the (changes of) actual values of the independent variables. 
In practice, several issues can arise: 

1. Variables on vastly different scales make exploratory data analysis more difficult; for example, 
   tools such as heatmap and visualizations like plots become harder to use. 
2. One variable can have much larger values than the others and it can dominate the cost function, in 
   which case the other variables wouldn't contribute to the fitting as much. 
3. Datasets containing continuous variables with large values and/or variance can make the 
   convergence of optimization algorithms take much longer. 

The idea, at a high-level, is to transform the column variables to put them on the same numeric scale; 
for example, between 0 and 1; -1 and 1 or between :math:`min` and :math:`max` for two constants, 
:math:`min, max`. However, care is needed for the following reasons: 

1. Not every pre-processing method is applicable to every variable/dataset. For example, variables 
   with a small number of very significant outliers can be skewed with techniques that use averages 
   while the structure of sparse variables would typically be lost if one attempted to center it at, 
   say 0. 
2. The parameters of a pre-processing step should be computed on **only the training** data (i.e., 
   after performing the train-test split) so as to not "leak" information from the test set. However, 
   it is very important to apply the pre-processing to the test set before predicting; otherwise, 
   the model performance will suffer. 

In addition to improving the overall modularity and reuse of our code, the ``Pipeline`` class will 
help in particular with point 2) above, as it will ensure the same pre-processing is applied to the
test data before predict is called. 


Data Standardization: Mean Removal and Variance Scaling 
--------------------------------------------------------

*Data standardization*, sometimes also referred to as *z-Score normalization*, is the process 
of transforming a continuous variable to have a mean of zero
and a standard deviation of 1. Mathematically, the procedure is straight-forward: for each 
continuous feature :math:`X_i` in the dataset, and each :math:`x \in X_i` we make the following 
update:

.. math::

  x \rightarrowtail (x - mean(X_i)) / std(X_i)

where:
 * :math:`mean(X_i)` is the mean of the column, :math:`X_i`
 * :math:`std(X_i)` is the standard deviation of the column, :math:`X_i`

It's clear that the updated values in the :math:`X_i` column have mean 0 and standard deviation 1. 

Applying data standardization to continuous columns in a dataset can be an important 
pre-processing step when the column variables have a normal distribution -- for example, not sparse,
and no significant outliers. 

*When to Use*: When the dataset is normally distributed (or close to it). 

``StandardScaler`` in sklearn 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``StandardScaler`` class from the ``sklearn.preprocessing`` module provides a convenience class
that implements data standardization. The classes in ``preprocessing`` module that perform 
transformations on data are all used in the following way:

1. Instantiate an instance of the class 
2. Fit the instance to the training data using the ``.fit()`` function. 
3. Apply the transformation to a dataset using the ``.transform()`` function. 

Note that we always apply the ``fit()`` to the **training data** to "learn" the scaling parameters 
(in this case the mean and standard deviation). We never apply it to test data, as this would 
cause our model to be fit in part based on the test data. 

.. warning:: 

    Using test data to fit the Scaler can lead to overly optimistic performance estimates. 
    A simple rule to remember is this: *Never call fit() on the test data.*

Let's see an example using our cars data set from Unit 1. Remember that we had created an 
updated version of the dataset that included our pre-processing. Let's load that one to avoid 
having to re-run all of the pre-processing steps: 

.. code-block:: python3 

    >>> cars = pd.read_csv('data/used_cars_data2.csv')
    >>> X = cars.drop(["Name", "Location", "Price"], axis=1)
    >>> y = cars["Price"]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # look at any particular column, e.g., Power: 
    >>> print(f"mean: {X_train['Power'].mean()}; std: {X_train['Power'].std()}")
    mean: 112.73469645700636; std: 52.7607709047988

We see that prior to standardization the Price column has a large mean and standard deviation.

.. code-block:: python3 

    >>> from sklearn.preprocessing import StandardScaler
    # step 1 -- Instantiate the Scaler
    >>> car_Scaler = StandardScaler()
    # step 2 -- fit the Scaler to the training data 
    >>> car_Scaler.fit(X_train)
    # step 3 -- apply the transformation; in this case, we apply it to the training data. 
    >>> X_train_scaled = car_Scaler.transform(X_train)

    >>> print(f"scaled mean: {X_train_scaled.mean()}; scaled std: {X_train_scaled.std()}")
    scaled mean: 9.601987668144666e-16; scaled std: 1.0

We see that the mean of the dataset after applying the transformation is (essentially) 0 
and the standard deviation is 1. 

Note that even though ``X_train`` was a DataFrame, ``X_train_scaled`` is an ndarray. If we try to 
use DataFrame indexing (e.g., ``X_train_scaled['Power']``) it will not work. We can of course get 
at specific columns using the column index. 

.. note:: 

    Even though the above method works fine, we recommend using the ``Pipeline`` class 
    described at the end of this module when combining data preprocessing with model 
    training. 


Robust Scalers 
---------------

When the dataset contains outliers that deviate significantly from the mean, using standardization
could result in worse performance because the outliers could dominate the mean/variance and crush the signal. 

In these cases, a robust Scaler based on different statistical methods, such as IQR, can be used instead. 
With a robust Scaler, the median is removed, and scaling is performed based on some percentage range. 

*When to Use*: When the dataset contains outliers that deviate significantly from the mean. 


``RobustScaler`` in sklearn 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``RobustScaler`` class in sklearn provides the same methods as the ``StandardScaler`` we just 
looked at. Just like before, we'll follow the following steps: 

1. Instantiate an instance of the class 
2. Fit the instance to the training data using the ``.fit()`` function. 
3. Apply the transformation to a dataset using the ``.transform()`` function. 

We'll look at an example of ``RobustScaler`` in the section on ``Pipelines``. For now, 
let's take a quick example involving a plain numpy array. 

.. code-block:: python3 

    # define a numpy array with an outlier --- most of the values are 
    # around 10, but there is one value of 10,000,000: 
    >>> n = np.array([10, 11, 9, 8, 8.5, 10000000, 9, 10, 10])

    print(n.mean(), np.median(n), n.std())
    1111119.5 10 3142693.8393535535

We see the that the mean and standard deviation are large, while the median is 10. 
Let's try scaling this array using both StandardScaler and RobustScaler. Note that 
we have to reshape the array to instruct the scalar that it should be treated as a single 
column feature (if it were a single sample consisting of multiple columns, we should reshape is 
with ``reshape(1, -1)``).

.. code-block:: python3 

    from sklearn.preprocessing import RobustScaler, StandardScaler
    std_scaler = StandardScaler().fit(n.reshape(-1,1))
    robust_scaler = RobustScaler().fit(n.reshape(-1,1))
    n_scaled_std = std_scaler.transform(n.reshape(-1,1))
    n_scaled_r = robust_scaler.transform(n.reshape(-1,1))

    print(n_scaled_std)
    print(n_scaled_r)


MaxAbs Scaler 
-------------

The last Scaler we will mention is the ``MaxAbs`` Scaler. 


*When to Use*: When the dataset contains sparse data. 


Pipelines 
---------

Also mention that every model has hypyerparameters and you can find them in the documentation. 

References and Additional Resources
-----------------------------------
