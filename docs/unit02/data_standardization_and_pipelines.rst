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
   say 0. More generally, the techniques we will look at in this module **apply to continuous 
   variables**. Categorical variables are often treated with different methods, such as 1-hot 
   encoding, which we have looked at previously. 
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
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
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
we have to reshape the array to instruct the scaler that it should be treated as a single 
column feature (if it were a single sample consisting of multiple columns, we should reshape is 
with ``reshape(1, -1)``).

.. code-block:: python3 

    from sklearn.preprocessing import RobustScaler, StandardScaler
    # 30 normally distributed points with mean 5 and std 3
    data = np.random.normal(5, 3, 20)
    df1 = pd.DataFrame({"data": data})
    print(df1.describe())

    # some outliers 
    outliers = np.array([150, 600, 900])
    df2 = pd.DataFrame({
        "data2": np.append(data, outliers)
    })
    print(df2.describe())

                data2
    count   23.000000
    mean    75.203711
    std    219.806640
    min     -4.457382
    25%      2.587355
    50%      5.318264
    75%      6.964271
    max    900.000000

Now, let's apply a robust scaler: 

.. code-block:: python3 

    robust_scaler = RobustScaler().fit(df2)
    robust_scaled_data = robust_scaler.transform(df2)


Let's see what these scalers did to the data: 

.. code-block:: python3 

    >>> robust_scaled_df = pd.DataFrame({"data": robust_scaled_data.reshape(-1)})
    >>> robust_scaled_data.describe()

                data
    count   23.000000
    mean    15.966825
    std     50.219529
    min     -2.233456
    25%     -0.623935
    50%      0.000000
    75%      0.376065
    max    204.409182

*Discussion:* Note that the range of values is still quite wide after applying the robust scaler. 
By comparison, what do you think would happen if we applied the ``StandardScaler`` to these data?

The range would be much more narrow. 


MaxAbs Scaler 
-------------

The last Scaler we will mention is the ``MaxAbsScaler``, short for "maximum absolute" scaler. 
This scaler uses the maximum absolute value of each feature to scale the values of that 
feature (i.e., the maximum absolute values of each feature after transformation will be 1). 
Note that itt does not attempt to shift/center the data, so if a feature is sparse 
(i.e., consists mostly of 0s), the data "spareness" structure will not be destroyed. 

Note also that this scaler does not reduce the effect of outliers. 


*When to Use*: When the dataset contains sparse data. 


Pipelines 
---------

The sklearn package provides a utility class called ``Pipeline`` that can be used 
to make your code more modular/reusable and to ensure that the same preprocessing 
steps are applied to training and test data in the appropriate way. 

The idea of the Pipeline is to define a sequence of transformations to preprocess 
data and fit the model. The intermediate steps can be any transformation that 
implement the ``Transforms`` API. 

There are a couple of ways of constructing ``Pipeline`` objects. The first way 
we will look at is with the ``make_pipeline()`` convenience function from the 
``sklearn.pipeline`` module. This method is good for simple pipelines where we don't 
need to refer to the attributes on objects within steps. Next, we will look at calling
the ``Pipeline()`` constructor (from the same module) directly. We will need to do this 
when we want to combine pipelines with ``GridSearchCV``, for example. 

An Initial Pipeline 
^^^^^^^^^^^^^^^^^^^^

Let's first build a pipeline to apply a scaler to the Pima Indians Diabetes dataset 
before fitting a KNN classifier model. In this first approach, we will hard code the 
number of neighbors, but we will see that the scaler already improves the performance. 

To begin, we will perform some initial data load and pre-processing. For backaround 
on this dataset in the pre-processing steps we took, see our 
KNN `lecture notes <knn.html#k-nn-in-sklearn>`_. 

.. code-block:: python3 

    data = pd.read_csv("../Diabetes-Pima/diabetes.csv")
    # Glucose, BMI, Insulin, Skin Thickness, Blood Pressure contains values which are 0
    data.loc[data.Glucose == 0, 'Glucose'] = data.Glucose.median()
    data.loc[data.BMI == 0, 'BMI'] = data.BMI.median()
    data.loc[data.Insulin == 0, 'Insulin'] = data.Insulin.median()
    data.loc[data.SkinThickness == 0, 'SkinThickness'] = data.SkinThickness.median()
    data.loc[data.BloodPressure == 0, 'BloodPressure'] = data.BloodPressure.median()

    # x are the dependent variables and y is the target variable
    X = data.drop('Outcome',axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

Recall from the notes that we found the optimal ``n_neighbors`` to be 13 using 
GridSearchCV in our previous lecture. We'll hard code the 13 value for now, but 
note that because we'll be using scaling, the optimal ``n_neighbors`` value could 
be different. 

To create a pipeline using the ``make_pipeline`` function, all we have to do is pass 
the objects (transformations) we want to perform as arguments in the order they 
should be performed. The last step of a pipeline should be the model to be fit. 

Here we create a pipeline with two steps: the ``StandardScaler`` and the 
``KNeighborsClassifier``: 

.. code-block:: python3 

    >>> pipe_line = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=13))

With the ``pipe_line`` object created, we now call ``fit()`` to execute each transformation 
in the pipeline. We pass the train dataset, just as we would when calling ``fit()`` on 
the transformation or model directly: 

.. code-block:: python3 

    >>> pipe_line.fit(X_train, y_train)

Finally, we call ``score()`` or a similar method to assess the model's performance. 
Note that the pipeline applies all of the transformations to the test data. This 
ensures we get optimal model performance. If we applied a scaling method to train the 
model but did not apply the same method to the test data, we wold likely get poor 
results. 

.. code-block:: python3 

    >>> print(pipe_line.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.    
    0.7532467532467533

Note that the score function uses accuracy by default here. Our model achieves 
75% accuracy on the test data. That's already an improvement over the model we learned 
without scaling (recall that we had achieved 71% previously).

Note also that the other methods are available, such as ``predict()``, on our 
``pipe_line`` object, so we can do things like: 

.. code-block:: python3 

    >>> from sklearn.metrics import classification_report
    >>> print(classification_report(y_test, pipe.predict(X_test)))


Pipeline with Named Steps and ``GridSearchCV``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We already saw some improvements with the simple pipeline above, but we can do better. 
We can search for the optimal hyperparameters (in our case, the ``n_neighbors``) 
given that the dataset has been scaled. 

To do that, we need to use the ``Pipeline`` constructor to name the steps of our 
pipeline. All we do is provide an additional argument, a string which is used for the  
name: 

.. code-block:: python3 

    from sklearn.pipeline import Pipeline

    p = pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('knn', KNeighborsClassifier()),
    ])

Here we have defined a pipeline with two steps, just as before. We named the first step
"scale" and the second one "knn". 
Note that we do not specify the ``n_neighbors`` value to the ``KNeighborsClassifier()``
constructor -- we're going to search for that. 


Now, we need to define our parameter grid, like we have done before, to describe the 
space of the parameters we want to search on. The key here is that we need to 
namespace the parameter by the step name, because a given parameter will only apply 
to a certain step. 

The way to do that is to use the step name, then two underscores (i.e., ``__``) 
and then the parameter name; i.e., ``<step_name>__<param_name>``. For example, 
``knn__n_neighbors`` refers to the ``n_neighbors`` attribute of the ``knn`` 
step. We then supply the range of values for the parameter just as before. 

Here is our ``param_grid`` definition: 

.. code-block:: python3 

    param_grid = {
        "knn__n_neighbors": np.arange(1, 100)
    }


With that, we can define the ``GridSearchCV`` object as before but this time 
passing the pipeline object instead of the model. We then call ``fit()`` and 
``score()`` etc., using the search object: 

.. code-block:: python3 

    search = GridSearchCV(p, param_grid, n_jobs=4)
    search.fit(X_train, y_train)
    print(f"Score with best parameters: {search.best_score_}")
    print(search.best_params_)    

    Score with best parameters: 0.7820872274143303
    {'knn__n_neighbors': 19}

Note that the optimal ``n_neighbors`` was 19, different from the optimal value of 
13 we found without the scaling, and the accuracy has increased to 78%. 


Pipeline With A Custom sklearn Model to Search Across Models
-------------------------------------------------------------

In this section, we provide an example of writing a custom model in sklearn. 
The idea is to allow us to search across models and hypyerparemeters within a 
single pipeline object. It also allows us to illustrate how relatively simple it 
is to extend the ``BaseEstimator`` class with custom behaviors. For more details, 
see [1]. 

We'll create a child class of the ``BaseEstimator`` class that accepts a model object 
as a parameter to the constructor and provides implementations of the ``fit()``, 
``predict()``, ``predict_proba()`` and ``score()`` methods that utilize the model. 
In this way, we will be able to pass the model object as a parameter in our param_grid 
attribute that will be used in the pipeline and search.

Here is the code for our class: 

.. code-block:: python3 

    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsClassifier

    class MultiModelClassifier(BaseEstimator):
        """
        A custom Estimator class that can be constructed with different model types. 
        For details on implementing custom Estimators, 
        see: https://scikit-learn.org/stable/developers/develop.html
        """

        def __init__(self, model=KNeighborsClassifier()):
            """
            A custom estimator parameterized by the model.
            Pass the result of an estimator constructor for `model`. By default, 
            it uses the KNeighborsClassifier().
            """
            self.model = model

        def fit(self, X, y=None, **kwargs):
            self.model.fit(X, y)
            return self
            
        def predict(self, X, y=None):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def score(self, X, y):
            return self.model.score(X, y)


You will see that the code is pretty straight-forward: in the constructor, all we do is 
save the model object that the user passed us as ``self.model``. Then, in each of the 
other methods, we simply call the corresponding method on ``self.model``. 

Let's see how to use this in a pipeline and grid search. First we define out pipeline. 
It will have two steps, the first one being the scaler and the second one the model. 
We'll use our new ``MultiModelClassifier`` as the model step. 


.. code-block:: python3

    p2 = Pipeline([
        ('scale', StandardScaler()),
        ('mmc', MultiModelClassifier()),
    ])


Now to define our parameter grid. This time, the ``param_grid`` object will be a 
list of dictionaries, with each dictionary corresponding to a parameter space to 
search over for a specific model. 

We define the model to use by setting the ``model`` parameter to the ``mmc`` step using the ``__``
notation. That is, ``"mmc__model"`` will be a key in our dictionary and will have a value 
which will be the model we want to use (but as a list -- all the keys should be lists).

Then, we can define the associated hyperparameters to search over for that model. 
Keep in mind that we will need two ``__`` since we will be referecing an attribute of the 
``model`` object within the ``mmc`` step. 
For example, we can put ``mmc__model__n_neighbors`` to refer to the ``n_neighbors`` 
hyperparameter of the ``mmc__model`` object when the model is ``KNeighborsClassifier``.
Here's a complete examples: 

.. code-block:: python3

    param_grid = [
        {
            "mmc__model": [KNeighborsClassifier()],
            "mmc__model__n_neighbors": np.arange(1, 100)
        },
        {
            "mmc__model": [RandomForestClassifier()],
            "mmc__model__n_estimators": np.arange(start=20, stop=150, step=3),
        },
    ]

We can now construct the search object, fit and score, as before: 

.. code-block:: python3 

    >>> gscv2 = GridSearchCV(p2, param_grid, cv=5)
    >>> gscv2.fit(X_train, y_train)
    >>> print("scaling best params: ", gscv2.best_params_)
    >>> accuracy_test2 = accuracy_score(y_test, gscv2.best_estimator_.predict(X_test))
    >>> print(f'Accuracy of best estimator WITH SCALING on test data is: {accuracy_test}')

    scaling best params:  {'mmc__model': RandomForestClassifier(), 'mmc__model__n_estimators': 62}
    Accuracy of best estimator WITH SCALING on test data is: 0.7359307359307359

The output indicates that the search found the RandomForestClassifier with 62 trees to perform 
best. 

.. note:: 

 Each of the models we have introduces have hyperparameters that can be tuned. 
 In some cases, we presented only a subset of those hyperparameters; in other cases, 
 we didn't mention any at all. This will purely because of time constraints. 
 We encourage you to explore the possible hyperparameters for each of the models 
 you work with by reading about them in the ``sklearn`` documentation. 

References and Additional Resources
-----------------------------------

1. Sklearn documentation: custom estimators. https://scikit-learn.org/stable/developers/develop.html
