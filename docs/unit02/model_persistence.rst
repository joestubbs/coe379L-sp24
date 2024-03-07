Model Persistence 
=================

In this short module, we introduce the notion of model persistence and why it is important, and 
we describe a simple method for persisting and reconstituting models to and from files saved on 
disk using the Python ``pickle`` module. 

By the end of this module, students should be able to: 

1. Understand the concept of model persistence and why it is important. 
2. Use the Python ``pickle`` module to persist and load models to and from a file. 
3. Understand the challenges and limitations of the ``pickle`` method. 


Introduction
-------------

Recall that, at a high level, the use of ML invovles the following process:

1. Find or collect raw data about the process or function.
2. Prepare the data for model training or fitting.
3. Train the model using some of the prepared data.
4. Validate the model using some of the prepared data.
5. Deploy the model to analyze new data samples.

We've look at pretty much all of these steps except for the last one which involves the topic 
of machine learning operations, or MLOps. In practice, we need a method for saving and deploying 
a model that has already been trained to an application where it can analyze new data. We 
certainly don't want to have to retrain the model every time we start our application, for several reasons: 

1. Model training requires data, which can be large and difficult to ship with our application. 
2. Training can be a time-consuming process. 
3. The training process might not be possible/reproducible on every device where we wish to deploy 
   our application. 

All of those reasons motivate the need to be able to save and load models that have already been trained. 

Here, we will look at a first method for saving and loading models to a file based on the 
Python ``pickle`` module, which is part of the standard library. The method we mention has the advantage that 
it is simple and can be used with many Python objects, not just models. However, it also comes with 
security risks, which we will cover. 


The ``pickle`` Module 
---------------------

The ``pickle`` module is part of the Python standard library and provides functions for serializing 
and deserializing Python objects to and from a bytestream. 

The process of converting a Python object 
to a bytestream is referred to as *pickling the object*, and the reverse process of taking a bytestream 
and converting it back to a Python object is called *unpickling*. 

Once a Python object has been converted to a bytestream with pickle, the bytestream can then be written 
to a file. Later, we can read the bytes back out of the file and reconstitute the original Python object. 

Many Python objects can be pickled, including the following: 

* builtin constants (True, False, None) 
* strings, bytes and bytearrays 
* *some* classes and class instances (specifically, the ones that implement ``__getstate__()``)
* lists, dictionaries, and tuples of picklable objects. 

In general, the models we have looked at from sklearn can be pickled. 

Conceptually, the ``pickle`` module is somewhat similar to JSON, providing a method for transmitting data 
to and from a specific format, but there are some key differences: 

* JSON is for text data, pickle can handle binary data (e.g., images, audio)
* JSON can be used in any programming language, while pickle can only be used with Python. 

Using the pickle module is straight-forward, and it provides a similar API to that of json. 
We use the following methods for serialization: 

* ``pickle.dumps(obj)`` converts the Python object ``obj`` to a bytestream. 
* ``pickle.dump(obj, file)`` converts the Python object ``obj`` to a bytestream and writes it to ``file``. 

And similarly, for deserializing:

* ``pickle.loads(bytes)`` converts the bytes object to a Python object. 
* ``pickle.load(file)`` reads the contents of ``file`` and converts the bytes to a Python.

Of course, the ``load()`` and ``loads()`` functions will fail if the bytes read in were not originally 
created by the pickle module. 

Let's see this in action. Suppose we have just trained a KNN classifier. 

.. code-block:: python3 

    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> knn.fit(X_train, y_train)

We can use pickle to save it to a file: 

.. code-block:: python3 

    import pickle 
    with open('my_knn_model', 'wb') as f:
        pickle.dump(knn, f)

Note the use of writing to the file in **binary format** (the ``'wb'`` flag in the call to ``open``). 
This is important --- the pickle output is a bytestream so without the ``b``, the write will fail. 

Now, we can read the model back in to a new Python object from the file. We can even shut down the 
Python kernel (i.e., exit the program) and restart it first. 

.. code-block:: python3 

    # load the model from disk: 
    with open('my_knn_model', 'rb') as f:
        model = pickle.load(f)    

Again, notice the use of reading the file in binary format. The load process will fail if we do not do 
that! 

But now, we can use ``model`` just as we would have used ``knn`` prior; we can go straight to predicting 
on test data (of course, if we shut down the kernel we will have to reimport the modules and redefine objects 
like ``y_test``): 

.. code-block:: python3 

    from sklearn.metrics import accuracy_score

    accuracy_test=accuracy_score(y_test, model.predict(X_test))
    print('Accuracy of loaded model from disk on test data is : {:.2}'.format(accuracy_test))   

    Accuracy of loaded model from disk on test data is : 0.68

.. note:: 

    Note that in general, Python callables (e.g., functions) *cannot* be pickled. If you need to serialize 
    a callable, consider using the third-party ``cloudpickle`` package instead, available from pypi [1].


A Word on Security with ``pickle``
-----------------------------------

We need to be very careful when using the ``pickle`` library to load Python objects. It is possible to 
serialize code that could harm your machine when loaded. For that reason, it is recommended that you 
**only** use ``pickle.load()`` and ``pickle.loads()`` on files and bytestreams that you know and trust 
(i.e., that you wrote yourself). As a result, ``pickle`` is not a suitable solution for some cases; 
for example, a web API or service that allows users to upload their own model and execute them on the 
cloud. Later, we'll look at some different techniques that can be used in these cases. 


.. warning:: 

    Never use pickle to load a bytestream that you did not write yourself. You could do harm to your 
    computer. 


Persisting Model Collections with Pickle and Checkpointing Large Searches 
-------------------------------------------------------------------------

Of course, we don't need to stop with just one model. Nothing prevents us from persisting an entire
collection of models in a Python object to single file. For example, we could use a dictionary, 
where the keys are the model type and the values are objects containing the metadata we care about. 

This approach could be quite valuable in the model training stage itself, not just in model deployment.
For example, suppose we are using GridSearchCV to search across a large hyperparameter space. It is 
quite possible the search to take many hours or even days, depending on the available compute power. 
In such a scenario, it may not be desirable or even possible to perform the entire search in one 
execution. Instead, one might take an approach like the following: 

1. Start a grid search program to begin the search.
2. As the search progresses, periodically write the intermediate results to a file using ``pickle``. 
   For example, if the search is considering multiple model types (Logistic Regression, Naive Bayes, 
   Decision Trees, etc.), the program could write the best result for each model type to the file once 
   it completes that search. 
3. Stop the program at any time, for example, when computing resources are not available. 
4. When the program starts back up again, the first thing it does is check the file to see 
   what models have already been searched through. It then crafts the grid search to pick up where 
   it last left off. 

The process above is called *checkpointing*, essentially, remembering work done previously and allowing 
a program to pick up where it left off. Checkpointing is very important for long-running processes. 

How might we implement checkpointing with model persistence? We won't give a complete solution but we 
will sketch the basic idea. 

First, we'll need functions to get and save the current state of our program. 

.. code-block:: python3 

    def get_state():
        # load the saved state from disk 
        with open("state", 'rb') as f:
            try:
                current_state = pickle.load(f)
            except:
                current_state = None

    def save_state(d):
        with open("state", 'wb') as f:
            pickle.dump(d, f)

We'll also need to provide a well-defined structure for the state dictionary. There are multiple 
ways to do this. Here, we simply save the "best" model object for each model type. 

.. code-block:: python3 

    def add_model_to_state(current_state, model_type, best_model):
        current_state[model_type] = best_model
        return current_state


Then, we'll need a way to create a param grid based on the current state: 

.. code-block:: python3

    def get_next_param_grid(current_state):

        # list of models we are interested in training 
        models = ["knn", "rf", "nb", "lr"]

        # full param gris that we want to search... 
        full_param_grid = {
          "knn": 
            {
                "mmc__model": [KNeighborsClassifier()],
                "mmc__model__n_neighbors": np.arange(1, 100)
            },
          "rf": 
            {
                "mmc__model": [RandomForestClassifier()],
                "mmc__model__n_estimators": np.arange(start=20, stop=150, step=3),
            },
        }

        for model in models:
            # if the model is already in the current state, then skip it -- we've already 
            # searched it previously. 
            if model in current_state.keys():
                continue 
            # otherwise, we've found the next param grid to search: 
            return full_param_grid[model]
        
        # terminating condition -- if all models have been trained, we're done 
        return None 
            

Then, our main program is a loop where we iteratively: 

1. Read the file
2. Get the next param grid 
3. Train and save the best fit model using GridSearchCV and the save_state function 

.. code-block:: python3 


    def main():
        while True:
            current_state = get_state()
            param_grid = get_next_param_grid(current_state)
            if param_grid is None: 
                break 
            train_and_save_param_grid(param_grid) # ToDo: implement...





References and Additional Resources
-----------------------------------

1. Cloudpickle Python Package on Github. https://github.com/cloudpipe/cloudpickle 
    


