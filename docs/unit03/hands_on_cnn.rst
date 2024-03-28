Hands-on CNN 
============================

We are given a dataset containing images of foods belonging to three categories: Bread, Soup and Vegetable-Fruits.
Our goal is to classify these images into their individual classes.

Step 0: Begin with getting the data on your machine.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a new terminal on your Jupyter and cd into /code/data folder. 
If you do not have data folder make one using ``mkdir``

.. code-block:: python3 

    mkdir data
    cd data

Inside the data folder wget the Food data

.. code-block:: python3 

    wget https://github.com/joestubbs/coe379L-sp24/raw/master/datasets/unit03/Food_cnn/Food_Data.zip


Once the Food_Data.zip file is downloaded, unzip it. Size of zip file is approximately 90MB

.. code-block:: python3 

   unzip Food_Data.zip

You should see three folders inside the Food_Data directory: Bread, Soup and Vegetable-Fruit

Next, create a python notebook ``outside`` the data directory. This is important for rest of the steps to work.

Step 1: Loading the data
~~~~~~~~~~~~~~~~~~~~~~~~~~
a) We need to make sure that train and test directories are empty to start with

.. code-block:: python3 

    # Let's make sure these directories are clean before we start
    import shutil
    try:
        shutil.rmtree("../data/Food-cnn-split/train")
        shutil.rmtree("../data/Food-cnn-split/test")
    except: 
        pass

``shutil`` is a module in Python's standard library that provides a higher-level interface for file operations. 
It's used for file and directory operations such as copying, moving, archiving, and more.

b) Lets create train and test directories for each of the three categories: Bread, Soup and Vegetable-Fruits

.. code-block:: python3 

    # We have three class which contains all the data: Bread, Soup and Vegetable-Fruit
    # Let's create directories for each class in the train and test directories.
    import os 
    # ensure directories exist
    from pathlib import Path

    Path("../data/Food-cnn-split/train/Bread").mkdir(parents=True, exist_ok=True)
    Path("../data/Food-cnn-split/train/Soup").mkdir(parents=True, exist_ok=True)
    Path("../data/Food-cnn-split/train/Vegetable-Fruit").mkdir(parents=True, exist_ok=True)

    Path("../data/Food-cnn-split/test/Bread").mkdir(parents=True, exist_ok=True)
    Path("../data/Food-cnn-split/test/Soup").mkdir(parents=True, exist_ok=True)
    Path("../data/Food-cnn-split/test/Vegetable-Fruit").mkdir(parents=True, exist_ok=True)

c) Next we need to collect all the paths for images in each category so we can split them in ``step d`` into train and test in a ratio 80:20 

.. code-block:: python3 

    # we need paths of images for individual classes so we can copy them in the new directories that we created above
    all_bread_file_paths = os.listdir('../data/Food_Data/Bread')
    all_soup_file_paths = os.listdir('../data/Food_Data/Soup')
    all_vegetable_fruit_file_paths = os.listdir('../data/Food_Data/Vegetable-Fruit')

d) Now we split the image paths into train and test by randomly selecting 80% of the images in train and 20% in test.
We also make sure there are no overlaps between the two splits.

.. code-block:: python3 

    import random

    train_bread_paths = random.sample(all_bread_file_paths, int(len(all_bread_file_paths)*0.8))
    print("train bread image count: ", len(train_bread_paths))
    test_bread_paths = [ p for p in all_bread_file_paths if p not in train_bread_paths]
    print("test bread image count: ", len(test_bread_paths))
    # ensure no overlap:
    overlap = [p for p in train_bread_paths if p in test_bread_paths]
    print("len of overlap: ", len(overlap))

    train_soup_paths = random.sample(all_soup_file_paths, int(len(all_soup_file_paths)*0.8))
    print("train soup image count: ", len(train_soup_paths))
    test_soup_paths = [ p for p in all_soup_file_paths if p not in train_soup_paths]
    print("test soup image count: ", len(test_soup_paths))
    # ensure no overlap:
    overlap = [p for p in train_soup_paths if p in test_soup_paths]
    print("len of overlap: ", len(overlap))

    train_vegetable_fruit_paths = random.sample(all_vegetable_fruit_file_paths, int(len(all_vegetable_fruit_file_paths)*0.8))
    print("train vegetable fruit image count: ", len(train_vegetable_fruit_paths))
    test_vegetable_fruit_paths = [ p for p in all_vegetable_fruit_file_paths if p not in train_vegetable_fruit_paths]
    print("test vegetable fruitimage count: ", len(test_vegetable_fruit_paths))
    # ensure no overlap:
    overlap = [p for p in train_bread_paths if p in test_bread_paths]
    print("len of overlap: ", len(overlap))

e) Next, we actually perform the copying of files in the train and test directories

.. code-block:: python3 

    # ensure to copy the images to the directories
    import shutil
    for p in train_bread_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Bread', p), os.path.join('../data/Food-cnn-split/train/Bread', p) )

    for p in test_bread_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Bread', p), os.path.join('../data/Food-cnn-split/test/Bread', p) )

    for p in train_soup_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Soup', p), os.path.join('../data/Food-cnn-split/train/Soup', p) )

    for p in test_soup_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Soup', p), os.path.join('../data/Food-cnn-split/test/Soup', p) )

    for p in train_vegetable_fruit_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Vegetable-Fruit', p), os.path.join('../data/Food-cnn-split/train/Vegetable-Fruit', p) )

    for p in test_vegetable_fruit_paths:
        shutil.copyfile(os.path.join('../data/Food_Data/Vegetable-Fruit', p), os.path.join('../data/Food-cnn-split/test/Vegetable-Fruit', p) )


    # check counts:
    print("Files in train/bread: ", len(os.listdir("../data/Food-cnn-split/train/Bread")))
    print("Files in train/soup: ", len(os.listdir("../data/Food-cnn-split/train/Soup")))
    print("Files in train/vegetable-fruit: ", len(os.listdir("../data/Food-cnn-split/train/Vegetable-Fruit")))

    print("Files in test/bread: ", len(os.listdir("../data/Food-cnn-split/test/Bread")))
    print("Files in test/soup: ", len(os.listdir("../data/Food-cnn-split/test/Soup")))
    print("Files in test/vegetable-fruit: ", len(os.listdir("../data/Food-cnn-split/test/Vegetable-Fruit")))

By the end of these steps, your train and test each should have 3 folders for Bread, Soup and Vegetable-Fruit populated.


Step 2: Data preprocessing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that we got the image files in train folder, we need to make sure they need pre-processing to be used for training models.
The images given to us of different sizes. We need to select a target size for each image (150,150,3), so the model can be trained on them.
We also need to Rescale the images by importing Rescaling from ``tensorflow.keras.layers.experimental.preprocessing``.
``Rescaling(scale=1./255)`` is used to rescale pixel values from the typical range of [0, 255] to the range [0, 1]. This rescaling is often used when dealing with image data to ensure that the values are within a suitable range for training neural networks.

We will use the ``tf.keras.utils.image_dataset_from_directory()`` function to create a TensorFlow tf.data.Dataset from image files in a directory. 
This will create a labeled dataset for us and the labels correspond to the directory that image is in.

Let's first install `tensorflow_datasets` so we can create train and validation datasets.

.. code-block:: python3

    pip install tensorflow_datasets --user 

You will have to restart Kernel from ``Kernel>Restart`` and continue from this step onwards.

.. code-block:: python3 

    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    train_data_dir = 'data/Food-cnn-split/train'

    batch_size = 32
    # target image size 
    img_height = 150
    img_width = 150

    # note that subset="training", "validation", "both", and dictates which dataset is returned
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )
    rescale = Rescaling(scale=1.0/255)
    train_rescale_ds = train_ds.map(lambda image,label:(rescale(image),label))
    val_rescale_ds = val_ds.map(lambda image,label:(rescale(image),label))

We will do a similar preprocessing on test data

.. code-block:: python3 

    test_data_dir = 'data/Food-cnn-split/test/'

    batch_size = 2

    # this is what was used in the paper --
    img_height = 150
    img_width = 150

    # note that subset="training", "validation", "both", and dictates what is returned
    test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    )

    # approach 1: manually rescale data --
    rescale = Rescaling(scale=1.0/255)
    test_rescale_ds = test_ds.map(lambda image,label:(rescale(image),label))

Now we have pre-processed datasets ``train_rescale_ds`` and ``val_rescale_ds`` and they are ready to be used for training the model.

Any Regular CNN 
~~~~~~~~~~~~~~~~~~~~~~~~~
We will build a CNN with 3 alternating convolutional and pooling layers and 2 dense hidden layers.
Output layer will have 3 classes and softmax activation function.

.. code-block:: python3 

    from keras import layers
    from keras import models
    import pandas as pd 
    from keras import optimizers

    # Intializing a sequential model
    model_cnn = models.Sequential()

    # Adding first conv layer with 64 filters and kernel size 3x3 , padding 'same' provides the output size same as the input size
    model_cnn.add(layers.Conv2D(?, (?, ?), activation='relu', padding="same", input_shape=(?,?,?)))
    
    # Adding max pooling to reduce the size of output of first conv layer
    model_cnn.add(layers.MaxPooling2D((2, 2), padding = 'same'))

    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model_cnn.add(layers.MaxPooling2D((2, 2), padding = 'same'))

    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model_cnn.add(layers.MaxPooling2D((?, ?), padding = 'same'))

    # flattening the output of the conv layer after max pooling to make it ready for creating dense connections
    model_cnn.add(layers.Flatten())

    # Adding a fully connected dense layer with 100 neurons    
    model_cnn.add(layers.Dense(100, activation='relu'))

    # Adding a fully connected dense layer with 84 neurons    
    model_cnn.add(layers.Dense(84, activation='relu'))

    # Adding the output layer with * neurons and activation functions as softmax since this is a multi-class classification problem  
    model_cnn.add(layers.Dense(?, activation='softmax'))

    # Compile model
    # RMSprop (Root Mean Square Propagation) is commonly used in training deep neural networks.
    model_cnn.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generating the summary of the model
    model_cnn.summary()

Let's fit the model and run it for 20 epochs.

.. code-block:: python3 

    #fit the model from image generator
    history = model_cnn.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds
    )

``Question: How will you compute accuracy on the test data?``

If you recall, we used evaluate() previously.

.. code-block:: python3 

    test_loss, test_accuracy = model_cnn.evaluate(test_rescale_ds, verbose=0)
    test_accuracy

We see the validation accuracy about 65% and test accuracy 73%. 

LeNet-5 
~~~~~~~~~~

We saw that LeNet-5 is a shallow network and has 2 alternating convolutional and pooling layers.
Let's try to train the LeNet-5 model on our training data.

.. code-block:: python3 


    from keras import layers
    from keras import models
    import pandas as pd 

    model_lenet5 = models.Sequential()
        
    # Layer 1: Convolutional layer with 6 filters of size 3x3, followed by average pooling
    model_lenet5.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(150,150,3)))
    model_lenet5.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # Layer 2: Convolutional layer with 16 filters of size 3x3, followed by average pooling
    model_lenet5.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model_lenet5.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # Flatten the feature maps to feed into fully connected layers
    ``which layer will you add here``

    # Layer 3: Fully connected layer with 120 neurons
    model_lenet5.add(layers.Dense(120, activation='relu'))

    # Layer 4: Fully connected layer with 84 neurons
    model_lenet5.add(layers.Dense(84, activation='relu'))

    # Output layer: Fully connected layer with num_classes neurons (e.g., 3 )
    model_lenet5.add(layers.Dense(3, activation='softmax'))

    # Compile model
    model_lenet5.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generating the summary of the model
    model_lenet5.summary()


Let's fit the model and run 20 epochs

.. code-block:: python3 

    #fit the model from image generator
    history = model_lenet5.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds
    )

We see even lower validation accuracy with this model and you might see high training accuracy, indicating overfitting.
There are techniques such as ``data-augmentation`` and adding ``Dropout`` layers to the model, to overcome overfitting. Time permitting we will disscuss them.

VGG16
~~~~~~~~~~

Let's now create a VGG16 model. For this we will use the pre-trained VGG16 model.

.. code-block:: python3 
   
    # Import VGG16 model from Keras applications
    from keras.applications.vgg16 import VGG16

    #Load the pre-trained VGG16 model with weights trained on ImageNet
    vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (150,150,3))
    vgg_model.summary()

    # Making all the layers of the VGG model non-trainable. i.e. freezing them
    for layer in vgg_model.layers:
        layer.trainable = False

    # Initializing the model
    new_model = models.Sequential()

    # Adding the convolutional part of the VGG16 model from above
    new_model.add(vgg_model)

    # Flattening the output of the VGG16 model because it is from a convolutional layer
    new_model.add(layers.Flatten())

    # Adding a dense input layer
    new_model.add(layers.Dense(32, activation='relu'))

    # Adding dropout prevents overfitting
    new_model.add(layers.Dropout(0.2))

    # Adding second input layer
    new_model.add(layers.Dense(32, activation='relu'))

    # Adding output layer
    new_model.add(layers.Dense(3, activation='softmax'))

    # Compiling the model
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Summary of the model
    new_model.summary()

    #fit the model from image generator
    history = new_model.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds,
    )

    test_loss, test_accuracy = new_model.evaluate(test_rescale_ds, verbose=0)
 
It turns out that this model gives us the best validation and test accuracy to solve the food classification problem.

