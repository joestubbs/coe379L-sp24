Project 03 - 25 Points
======================

**Date Assigned:** March 26, 2024

**Due Date:** Thursday, April 11, 2024, 5 pm CST.

**Group Assignment:** Students can work individually or in groups of two on this assignment. 
When working in groups, we expect both students to contribute equally to all aspects of the 
project. You are allowed to talk to students in other groups about the project, but 
please do not copy any code for the notebook or text for the report.

If you use ChatGPT, please state exactly how you used it. For example, state which parts of the 
code it helped you generate, which errors it helped you debug, etc. Please do not use ChatGPT to 
generate the report for part 3. 

**Late Policy:**  Late projects will be accepted at a penalty of 1 point per day late, 
up to five days late. After the fifth late date, we will no longer be able to accept 
late submissions. In extreme cases (e.g., severe illness, death in the family, etc.) special 
accommodations can be made. Please notify us as soon as possible if you have such a situation. 

**Project Description:**
You are given a dataset which contains satellite images from Texas after Hurricane Harvey. 
There are damaged and non-damaged building images organized into respective folders. 
You can find the project 3 dataset 
on the course GitHub repository 
`here <https://github.com/joestubbs/coe379L-sp24/tree/master/datasets/unit03/Project3>`_. 

Your goal is to build multiple neural 
networks based on different architectures to classify images as containing buildings that 
are either damaged or not damaged. You will evaluate each of the networks you develop and 
produce and select the "best" network to "deploy". Note that this is a **binary classification**
problem, where the goal it to classify whether the structure in the image **has damage** or 
**does not have damage**. 

**Part 1: (3 points)** Data preprocessing and visualization

You will need to perform data analysis and pre-processing to prepare the images for training. 
At a minimum, you should:

a) Write code to load the data into Python data structures 
b) Investigate the datasets to determine basic attributes of the images
c) Ensure data is split for training, validation and testing and perform any additional 
   preprocessing (e.g., rescaling, normalization, etc.) so that it can be used 
   for training/evaluation of the neural networks you will build in Part 2. 

**Part 2: (10 points)** Model design, training and evaluation

You will explore different model architectures that we have seen in class, including: 

a) A dense (i.e., fully connected) ANN
b) The Lenet-5 CNN architecture
c) Alternate-Lenet-5 CNN architecture, described in paper/except 
   (Table 1, Page 12 of the research paper https://arxiv.org/pdf/1807.01688.pdf, but note 
   that the dataset is not the same as that analyzed in the paper.)

You are free to experiment with different variants on all three architectures above. 
For example, for the fully connected ANN, feel free to experiment with different numbers 
of layers and perceptrons. Train and evaluate each model you build,and select the "best" 
performing model.

Note that the input and output dimensions **are fixed**, as the 
inputs (images) and the outputs (labels) have been given. These have important implications for your 
architecture. Make sure you understand the constraints these impose before beginning to design and 
implement your networks. Failure to implement these correctly will lead to incorrect architectures 
and significant penalty on the project grade. 

**Note:** You can also try to run the VGG-16 architecture from class, however, you may run
into long runtimes and/or memory limits on the VM. It is also possible, depending on the 
architecture that you choose, that you could also run into memory constraints with any of the 
other architectures. If you are hitting memory issues, you can try to decrease the ``batch_size``
parameter in the ``.fit()`` function, as described in the notes. 


**Part 3: (7 points)** Model deployment

For the best model built in part 2, persist the trained model to disk so that it can be 
reconstituted easily. 
Develop a simple inference server to serve your trained model over HTTP. There should be 
at least two endpoints:

a) A model summary endpoint providing metadata about the model
b) An inference endpoint that can perform classification on an image. Note: this 
   endpoint will need to accept a binary message payload containing the image to 
   classify and return a JSON response containing the results of the inference. 

Package your model inference server in a Docker container image and push the image to the 
Docker Hub. Provide instructions for starting and stopping your inference server using 
the docker-compose file. Provide examples of how to make requests to your inference server. 

**Bonus:** We will evaluate each of the model inference servers submitted against 
a reserved dataset. The top three models will get bonus points as follows:

* 1st place: 2 points 
* 2nd place: 1 point 


**Part 4: (7 points)** Write a 3 page report summarizing your work. 
Be sure to include something about the following:

* Data preparation: what did you do? (1 pt)
* Model design: which architectures did you explore and what decisions did you make for 
  each? (2 pts)
* Model evaluation: what model performed the best? How confident are you in your model? (1 pt)
* Model deployment and inference: a brief description of how to deploy/serve your model 
  and how to use the model for inference (this material should also be in the 
  README with examples) (1 pt)


Submission guidelines: Part 1 and Part 2 should be submitted as one notebook file. 
Part 3 should include a Dockerfile, a docker image (prebuilt and pushed to Docker Hub) and 
a docker-compose.yml file for starting the container. It should also include a README with 
instructions for using the container image, docker-compose file and example requests. 
Part 4 should be submitted as a PDF file. 


**In-class Project Checkpoint Thursday, April 4th**. We will devote the first portion of Thursday's 
class to checking in on the project and answering questions. 