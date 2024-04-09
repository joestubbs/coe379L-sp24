Hands-on Transformers 
=====================

In this module, we introduce the ``transformers`` library from Hugging Face and we show some 
initial examples of working with pre-trained models for solving NLP tasks. 

By the end of this module, students should be able to: 

1. Use pipeline objects to work with pre-trained models on a number of NLP tasks.
2. Understand the basic components of a pipeline, including data preprocessing, 
   model application and post-processing.
3. Understand how to use a Tokenizer to convert text to numerical inputs and to 
   convert numerical values back to text. 

Introduction to Pipelines 
-------------------------
As we mentioned briefly in the previous module, ``pipeline`` objects from the transformers library 
are basic abstractions that simplify the interaction with large models. In general, the following
steps must be taken to perform inference with a model on some input text: 

1. Convert the raw text to tokens (i.e., *input ids*) using a *tokenizer*.
2. Apply the model to the input ids to produce *logits*, that is, raw numeric values.  
3. Post-process the outputs of the model to produce probabilities (e.g., through the application 
   of *softmax*) and then class labels. 

These high-level steps are depicted in the diagram below: 

.. figure:: ./images/HF_pipeline.png 
    :width: 500px
    :align: center

    The basic components of a pipeline. 
    (Image credit: HuggingFace NLP Course: Behind the Pipeline [1])

Each step involves multiple complexities that we will explain. But first, it is worth pointing 
out that the transformers ``pipeline()`` function can already be used to solve common NLP tasks 
without any additional complexity. 

The simplest way to create a pipeline is to use the ``pipeline`` function and pass a specific 
task type. For example, we can create a pipeline by specifying the English to French translation 
task type, ``translation_en_to_fr``:

.. code-block:: python3

    from transformers import pipeline
    en_to_fr_translator = pipeline("translation_en_to_fr")

This bit of code first looks up the default model for this task type and checks whether that model 
has already been downloaded to you huggingface cache directory. If not, it downloads it to the 
cache directory and then instantiates the model. 

(Note that, by default, the huggingface cache directory is ``~/.cache/huggingface``, but you can 
change that by setting the ``$HF_HOME`` environment variable). 

We can now pass it some input and directly get some output: 

.. code-block:: python3 

    en_to_fr_translator("Hello, my name is Joe.")
    -> [{'translation_text': 'Bonjour, mon nom est Joe.'}]

We can pass it multiple inputs as well, as a Python list: 

.. code-block:: python3 

    en_to_fr_translator(["Machine learning is a branch of Artificial Intelligence", 
                     "The United States Declaration of Independence was written in 1776."])
    -> [{'translation_text': 'L’apprentissage automatique est une branche de l’intelligence artificielle'},
        {'translation_text': "La Déclaration d'indépendance des États-Unis a été rédigée en 1776."}]


The transformers library includes many recognized tasks with default models. There are task types 
from each of the following areas:

* Computer Vision, including text-to-image, image-to-text, and image-to-image tasks.
* Natural Language Processing, including sentiment analysis, language translation and question answering
  tasks. 
* Audio processing, including text-to-speech, audio classification and speech recognition tasks. 
* Multimodal, including document question and answering (i.e., answering questions on a visual document) 
  and visual question and answering (answering open-ended questions based on an image). 

Some specific examples include:

* ``translation_xx_to_yy`` -- Language translation from language xx to language yy.
* ``sentiment-analysis`` -- Also called text classification, i.e., classifying the sentiment 
  expressed in text. 
* ``summarization`` -- Producing a summary of the input text. 
* ``image-classification`` -- Classifying objects in an image. 
* ``image-to-text`` -- Generate a summary/caption for an image. 
More information is available on the HuggingFace documentation site, here. [2].

And just as with the language translation pipeline we defined above, we can defined similar 
pipelines for other tasks. For example, a text summarization pipeline: 

.. code-block:: python3 

    summarizer = pipeline("summarization")
    summarizer("""NLP is one of the oldest areas of AI and has a long history dating back at least to the 1950s.
        One of the first efforts to garner public attention was the Georgetown-IBM experiment in 1954, which
        attempted automatically translate Russian sentences to English.
        Here is a screenshot from an early, famous NPL program called ELIZA, developed at MIT between 1964 and
        1967. THe ELIZA program prompted users with questions in natural language text and enabled them to
        submit answers, also in natural language. The goal was to simulate a psychotherapy session.""")
    
    Output ->
    [{'summary_text': " NPL is one of the oldest areas of AI and has a long history dating back at least to 
      the 1950s . The Georgetown-IBM experiment in 1954 attempted to automatically translate Russian sentences 
      to English . MIT's ELIZA program prompted users with questions in natural language text and enabled 
      them to answer them with answers ."}]

Some tasks, however, do not have a default model. For example, if we try to build a pipeline for the 
English to Spanish translation task, we get an error: 

.. code-block:: python3 

    en_to_es_translator = pipeline("translation_en_to_es")

    -> ValueError: The task does not provide any default models for options ('en', 'es')

There are, however, models for English to Spanish translation are available from the transformers 
library. How do we go about finding them? One option is to use the HuggingFace Hub to search 
for models by task. The transformers library can utilize any of the publicly available models on 
the hub. 

1. Navigate to the HuggingFace website, `here <https://huggingface.co/>`_. 
2. Click Models to browse and search for models. As of the time of this writing there are 
   over 595,000 models on the hub. 
3. Click to filter by task type; we would like to search for models that can perform the 
   "Translation" task type, so we click that. 
4. Next, select the "Languages" filter tab to filter by languages. We are interested in English to 
   Spanish, so we select those. 

.. figure:: ./images/HF_Hub_1.png
    :width: 700px
    :align: center

In the screenshot above we see.. 

.. code-block:: python3 

    en_sp_translator = pipeline(model="Helsinki-NLP/opus-mt-en-es")
    en_sp_translator("Hello, my name is Joe.")
    -> [{'translation_text': 'Hola, mi nombre es Joe.'}]


Tokenizers
----------


Additional References 
---------------------
1. HuggingFace NLP Course. Chapter 2: Behind the Pipeline. https://huggingface.co/learn/nlp-course/chapter2/2
2. HuggingFace Tasks. https://huggingface.co/tasks



