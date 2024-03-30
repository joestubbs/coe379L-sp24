MLOps
=====

In this module, we introduce concepts and techniques related to MLOps, that is, 
the automation of operations involved in ML pipelines for applications. By the 
end of this module, students should be able to:

* Describes the basics of the entire ML application workflow and lifecycle. 
* Implement deployment automation for an arbitrary ML model using Python, Flask, and Docker. 
* Implement deployment automation for Tensoflow models using Tensorflow serving. 


The ML Lifecycle and MLOps 
---------------------------

Just like any other software, applications developed with machine learning are 
products that evolve over time. We repeatedly update the software as we gather more data, 
develop better models and generally make improvements. 

MLOps is a modification of the concept of DevOps, a set of techniques and best practices 
that originated in the late 2000s with the organization of DevOpsDays (2009) and related 
events. DevOps is a combination of the terms *software development* and *IT operations*, 
and the idea was to shorten the time between when new software features were developed 
(i.e., new code being written) and when those features would be released and available 
to users. In order to shorten that time window, a number tasks needed to be automated, 
including:

* Building/compiling software binaries, and/or assembling software packages. 
* Running tests on the new software to ensure it was high quality. 
* Integrating the software with other components in the ecosystem (e.g., other microservices
  in a microservice architecture) and running integration tests across the entire system. 
* Deploying the tested code to production. 


MLOps modifies the concept to DevOps concept to accommodate the processes involved with 
building and deploying new ML applications. At a high-level, the life cycle consists of: 

1. *Data collection and preprocessing:* as we have seen, large amounts of high-quality 
   data is essential to training models that can make accurate predictions. 
2. *Model training:* Once we have gathered and preprocessed data, we train our model. 
   As we have seen, this can involve lengthy executions to search across different model 
   types and hyperparameter spaces. 
3. *Model evaluation and validation:* As we train new versions of our model, we evaluate 
   them using various metrics (e.g., accuracy, precision, recall, F1, etc.). Beyond 
   evaluating models against specific metrics, a number 
   of additional validation steps can be taken, including: *automated testing*, to ensure 
   the application works as intended on important and/or edge cases; *fairness/bias assessment*, 
   for example, by evaluating your model against a separate hold out dataset that is specifically 
   engineered to ensure that it is representative and inclusive of all perspectives. 
4. *Model deployment:* after a model version has been validated, we are ready to package and 
   deploy it, either to a pre-production environment such as a test of QA environment, or to 
   production. 
5. *Integration Testing, Acceptance Testing, and other automated testing:* typically, the trained 
   mode is first deployed to a test or QA environment where automated tests are run. 
   Various kinds of tests could be run, including integration tests (to test the ML model's 
   integration into the rest of the application), functional or acceptance testing, to evaluate 
   high level functions of the application, and performance and/or load testing, to ensure the application 
   perfoms well under various loads. If the testing produces acceptable results, the model is 
   then deployed to production. 
6. *Production Monitoring:* once an ML model has been deployed to production, it must be monitored 
   just like any other application component. The model will see new data samples in production, 
   and these should be collected and evaluated. Various changes in the environment, referred to as 
   *drift* can cause model performance to degrade. For example, some facial recognition systems 
   saw preformance degradation during the pandemic as a result of people wearing masks. Language 
   models see performance degradation over time due to the introduction and use of new words.  

.. figure:: ./images/MLOps.png
    :width: 800px
    :align: center

    The ML Application Development and Operations Lifecycle

So far in this course, we have mostly focused on 1), 2) and 3). Below, we  on specific techniques for
Model Deployment. 

Model Deployment 
-----------------
There are a number of considerations when planning a model deployment. At a minimum, the software must 
be packaged and delivered in a way that allows it to be utilized by the rest of the application. 


Inference Server 
^^^^^^^^^^^^^^^^
The concept of an *inference server* has gained traction in the ML community. The idea is to wrap the 
trained ML model in a lightweight server that can be executed over the network. Commonly, this is done 
either as an HTTP 1.x/REST API or an HTTP 2/gRPC server. 

For example, a REST API inference server for the model we developed to classify images with food objects 
may have the following endpoint: 

+---------------------------------+------------+---------------------------------------------+
| **Route**                       | **Method** | **What it should do**                       |
+---------------------------------+------------+---------------------------------------------+
| ``/models/food/v1``             | GET        | Return basic information about v1 of model  |
+---------------------------------+------------+---------------------------------------------+
| ``/models/food/v1``             | POST       | Classify food object in image payload using |
|                                 |            | version 1 (v1) of the model.                |
+---------------------------------+------------+---------------------------------------------+

When a client makes an HTTP POST request to ``/models/food/v1`` they send an image as part of the 
payload. The inference server must:

1. Retrieve the image out of the request payload. 
2. Perform any preprocessing necessary on the image byte stream. 
3. Apply the model to the processed image data to get a classification result. 
4. Package the classification result into a convenient data structure (e.g., JSON).
5. Send a response with the classification data structure included as the message body.  

As you can see, we have encoded both the kind of model ("food") as well as the version ("v1") into 
our URL structure. This means that if we developed another model, for example, our handwritten digits 
classifier, we could easily add it to our inference server. We could also easily add a new version of 
the food model and serve both at the same time. 

There are a number of advantages to using an inference server architecture, many of which are just the 
advantages enjoyed by all HTTP/microservice architectures: 

1. *Framework agnostic:* Regardless of which ML framework your model is developed in, it can be packaged 
   into an inference server. With that said, some solutions are framework-specific. In fact, one of the 
   solutions we'll look at is Tensorflow Serving, which serves Tensorflow models (and other kinds of 
   *servables*). 
2. *Language agnostic API:* Components of the application can interact easily with the inference server, 
   regardless of the programming language they are written in, because all modern languages have an HTTP 
   client. 
3. *Scalability:* Multiple components of the application can interact with the model inference server, 
   even from different computers. Additionally, multiple instances of the inference server itself can 
   be deployed to increase the 
4. *Plug-and-play and model chaining:* The concept of *plug-and-play* for ML models is the idea or goals
   of enabling different models to be "plugged" into an application with little to no code changes to 
   the rest of the application. In order to achieve this, different models that perform the same (or similar)
   task must conform to a common interface. An HTTP interface is one possible mechanism. Similarly, 
   *model chaining* is the idea that we can feed outputs of one model as inputs to another model. For example,
   we may have one model that finds language characters in an image and another model that translates 
   words from one language to another (for example, 
   `Google image translate <https://support.google.com/translate/answer/6142483?hl=en&co=GENIE.Platform%3DDesktop>`_). 
   If individual models use HTTP requests and responses, the responses from one model can be easily fed into 
   as a request to the next model. 
5. *Versioning:* There are multiple, intuitive ways to version a model inference server. One which was suggested 
   above is to use the URL to encode the version. These methods will be familiar to most developers, as REST 
   APIs (and HTTP services more generally) have become common in cloud computing. 

What do we need to build an ML inference server? The basic ingredients are as follows: 

1. *Serialize and deserialize trained models* --- we saw how to do this with sklearn, but we will quickly 
   see how to do this with Keras. 
2. *Write the inference server code* --- we will see two methods for doing this, including a "generic" 
   method using flask and a Tensorflow-specific method (Tensorflow Serving)
3. *Package the server as a docker container image* --- This will simplify deployment and make our server 
   more portable. 
4. *Deploy the server as a container* --- We can use a simple script, docker-compose, or something more 
   elaborate such as Kubernetes. 


Serializing models
^^^^^^^^^^^^^^^^^^^

Additional References
----------------------
1. Machine Learning Systems with TinyML. Chapter 14: Embedded AIOps. https://harvard-edge.github.io/cs249r_book/contents/ops/ops.html#key-components-of-mlops
