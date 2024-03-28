Introduction to Transformers 
=============================

In this first module, we introduce the Transformer architecture and cover the main 
ideas of *attention* as presented in the seminal 2017 paper, "Attention Is All You Need" [1].
By the end of this module, students should be able to:

1. Understand..
2. 
3. 

.. warning:: 

    The topics in this unit are very new and evolving very quickly! It is quite possible 
    or even likely that these materials will become outdated in the near future.  

Background on Natural Language Processing (NLP)
----------------------------------------------
The transformer architecture that we will introduce in the next section was originally built to 
deal with natural language processing (NLP) tasks, specifically the task of language translation;
that is, translating text from English to Spanish or from Russian to French, etc. In general,
NPL focuses on tasks involving computer understanding of text data, such as that in books, 
articles, web pages, social media posts, etc. Some common NLP tasks include the following: 

1. *Sentiment Analysis:* what is the sentiment expressed by the text? For example, does the author 
   express a favorable or unfavorable opinion of a book, article, website, product, etc.? 
2. *Text classification:* for example, classifying a word by part of speech (e.g., noun, verb, adjective), 
   a book or article by topic (mathematics, computer science, biology), etc. 
   Sentiment analysis can be thought of a special case of text classification where we are classifying the 
   sentiment expressed into two classes (*favorable* and *unfavorable*). 
3. *Text generation:* Filling in the end of a sentence (e.g., autocomplete), filling in masked/blanked out 
   words within sentences, generating entire new sentences from a prompt. 
4. *Language translation:* translating a text from English to French or from Russian to Spanish, etc. 
5. *Question and Answer:* Providing answers to questions posed in natural language; e.g., Question: *"Who was the 
   first president of the United States?"* Answer: *"George Washington"*.

NLP is one of the oldest areas of AI and has a long history dating back at least to the 1950s. 
One of the first efforts to garner public attention was the Georgetown-IBM experiment in 1954, which 
attempted automatically translate Russian sentences to English.

.. There have been a number of instances in the past where bold claims did not come to fruition. For example,   
  the Georgetown-IBM experiment in 1954 involved work and a demonstration to automatically translate 
  Russian sentences to English. The scientists claimed at that time that automatic language translation 
  would be solved by machines within 3 to 5 years. 

Here is a screenshot from an early, famous NPL program called ELIZA, developed at MIT between 1964 and 
1967. THe ELIZA program prompted users with questions in natural language text and enabled them to 
submit answers, also in natural language. The goal was to simulate a psychotherapy session. 

.. figure:: ./images/ELIZA.png
    :width: 800px
    :align: center

ELIZA wwas able to resemble human-like behaviors on occasion, though its practical use was relatively 
limited.

In the 1970s, NPL researchers introduced the notion of *ontologies*, that is, formally structured and 
controlled vocabularies for specific topics or areas. It was during this time that the first chatbot 
programs were written. In the early 1970s, the chat program PARRY was developed and hooked up to 
ELIZA resulting in the following dialog. 

.. figure:: ./images/PARRY_ELIZA_1.png
    :width: 300px
    :align: left

.. figure:: ./images/PARRY_ELIZA_2.png
    :width: 300px
    :align: right


In the 1980s and 1990s, statistical methods began to be used on NLP tasks, with some success. 
However, with the growth of the internet and available data, these methods were
overshadowed by artificial neural networks and ultimately deep learning models trained on 
large amounts of data. 

A Prelude to Transformers: Sequential Data and RNNs [1]_
--------------------------------------------------------

In 2017, a group of researchers at Google Research introduced a new deep neural architecture 
called Transformer in a paper called "Attention Is All You Need" [1]. In that paper, the 
focus was on natural language processing (NLP) and specifically, language translation. 
Up to that point, Recurrent Neural Networks (RNNs) were considered state-of-the-art for 
language translation, and the paper introduced a key idea, *attention*, to address some 
shortcomings in RNNs. To gain a basic understanding of the key concepts of the transformer 
model, we'll need to review some background on sequential data and RNNs, which we can think of 
as an effort to enable enable neural networks to learn patterns in sequential data. 

Sequential Data 
^^^^^^^^^^^^^^^^
Sequential data, also sometimes called temporal data, is just data that contains an ordered or 
temporal structure. There are many types of sequential data all around us; for example: 

* The individual words within a text of natural language. 
* The position of a moving object or projectile. 
* The temperature of a location, as a function of time. 
* Stock prices as a function of time. 
* Medical signals (heart rates, EKGs)

And the key point is that, to whatever extent these data exhibit patterns, the patterns will depend on 
part on ordering of the events. For example, we know that the order in which words appear can have a 
big impact on the meaning. Consider two sentences: 

* The food was good, not bad at all! 
* The food was bad, not good at all! 

These two sentences have opposite meaning even though they are are comprised of the same 8 words!

* all, at, bad, food, good, not, the, was 

Similarly, is we are trying to predict the position of a moving object or the value of a stock 
at a given time *t*, we will have a difficult time if we are not given information about the values 
at previous times. On the other hand, we do expect the values at a given time to be, at least in part, 
determined by the values at previous times. 


Neurons with Recurrence
^^^^^^^^^^^^^^^^^^^^^^^
How should we try to go about modelling sequential data in a neural network? 
Recall our notion of a perceptron and feedforward 
network from Unit 3. There was no notion of sequential data there. There were just inputs on the left 
and outputs on the right. 

.. figure:: ./images/ann-arch-overview.png
    :width: 1000px
    :align: center

How might we modify that architecture to capture the notion of sequence? One idea is depicted 
below. If we think of a single, feedforward network as predicting the output at a given time, *t*, then 
we can essentially use a set of networks, stacked side by side, with each individual network used to 
compute the output based on the input at a given time step. 

Of course, our goal with sequential data is to allow the network to learn patterns in the data across 
time steps. If we just had individual networks for each time step that were not connected, we wouldn't 
be able to achieve our goal. 

This is where RNNs and the notion of a recurrence relation comes in; the idea is to feed the output of 
the network at a given time step as an additional input into the network handling the next time step, 
along with the input, *x*, at that next time step. 

First: a quick digression to recall the idea of a recurrence relation. 
Let :math:`s_1, s_2, ..., s_n, ...` be a sequence of numbers. 
Recall from mathematics that a *recurrence relation* is just an equation that expresses each element 
of a sequence as a function of one or more preceding elements in the sequence.

.. math:: 

    s_n = f(s_{n-1}, s_{n-2}, ..., s_{n-k})

For example, the famous Fibonacci sequence is given by the simple recurrence relation: 

.. math:: 

    (1)\;\;\;\;  F_n = F_{n-1} + F_{n-2} 

with :math:`F_0 = 0` and :math:`F_1 = 1`. Repeated application of the equation :math:`(1)`, gives 
the familiar values: 

.. math:: 

    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...


Coming back to the task at hand of learning patterns across time steps in sequential data, the 
basic idea is to pass the output from one time step as an additional input to the 
layer for the next time step. This is depicted in the following diagram: 

.. figure:: ./images/RNN.png
    :width: 1000px
    :align: center

Write :math:`h=h_t` for the intermediate output signal at time step *t* that is passed as input 
to the next time step. 
Then we can write :math:`y_t = f(x_t, h_{t-1})` where `f` represents the neural network 
depicted above. 

Furthermore, we can make the assumption that the sequence :math:`h_t` conforms a recurrence relation
and similarly write 

.. math:: 
    
    h_t := f(x_t, h_{t-1})
    
That is, the neural network is also responsible for computing the intermediate output state 
from the previous states. The individual values :math:`h_t` can 
be thought of as the "memory state" of the network at time step *t*, i.e., the neural network 
"remembering" outputs from previous time steps. 

We can also think of the RNN as being implemented using a loop, iteratively computing the intermediate
outputs, :math:`y_t`, from the inputs :math:`x_t` and the memory state, :math:`h_{t-}`. We depict an 
example pseudo code implementation below: 

.. code-block:: python 

    # pseudo code of an RNN implementation in Python...
    rnn = RNN() 

    # initialize the memory states to 0s
    h = [0, 0, 0, 0, ... , 0]

    # the input sequence of words 
    sentence = ["Let's", "predict", "the", "next", "word", "in", "this"]

    # basic RNN implementation is just a loop, passing each word in the sentence as well as 
    # the "memory" state into itself each time.. hence, "recurrence"  
    for word in sentence:
        prediction, h = rnn(word, h)
    
    # get the final prediction
    print(prediction)
    >>> "sentence"

Limitations of RNNs 
^^^^^^^^^^^^^^^^^^^^
While RNNs were able to achieve state-of-the-art performance on some NLP tasks, they ultimately exhibited some 
fundamental limitations:

1. *Limitations on memory:* RNNs require that sequential information is encoded and passed in, 
   time step by time step. 
   This creates a challenge when dealing with long input sequences, where the outputs depend on 
   inputs appearing early in the sequence. Think, for example, of translating an entire book in 
   one language to another, where knowledge of characters introduced in an early part of the book 
   is needed for translating parts at the end. 

2. *Slow due to lack of parallelism:* Again, because RNNs process one input at a time, they 
   cannot take advantage of parallelism for speed up, and this makes them slow. 

As a result of the two shortcoming above, RNNs have not able to handle sequences with 10s or 100s of thousands 
of items. 


Foundations of Transformer Architecture
---------------------------------------
As mentioned previously, the Transformer architecture, initially presented in a paper from 2017, 
was at least in part an attempt to overcome some of the limitations of RNNs. The paper, entitled 
"Attention Is All You Need" made famous the notion of *attention*, and it combined this idea with 
other ideas to formulate a new deep network architecture. We will cover the basics of these 
ideas without treating all of the technical details. 


.. figure:: ./images/Attention_is_all_you_need.png
    :width: 800px
    :align: center


Overview of the Transformer Architecture 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The transformer architecture as presented in the original "Attention Is All You Need" paper is depicted 
below. There are two primary components in the architecture: an *encoder*, depicted on the left half, 
and a *decoder*, depicted on the right half. You will notice that the two halves are almost identical, 
with the decoder adding just one additional component called the *Masked Multi-head Attention* instead 
of the plain (i.e., unmasked) multi-head attention.  

Thus, if we just focus on one side of the architecture, the primary components (from bottom to top) 
are as follows:

* The language embedding 
* The attention component 
* The feed forward network 

Note that the recurrence relation has been removed and the sequential input data is fed in all at once. 
This is the major change introduced by Transformer over RNN. 

.. figure:: ./images/Transformer_arch.png
    :width: 500px
    :align: center

We'll look at each of these primary components to try and build some intuition behind what they are doing. 
We'll start with the attention component, as it could be considered the most important. 

Intuition Behind (Self-)Attention 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The goal with attention is to focus on the most important features for whatever task is at hand. 
Said differently, we want a mechanism that enables the model to selectively focus on specific parts 
of an input sequence. 

For example, for the task of object detection in an image, where we want to determine if an object 
contains a human face, certain features, such as the eyes, nose, mouth, and hair, are arguably 
the most important parts of the input for the task. 
And if you think about it, this is exactly how your brain would determine if an image contained a face 
--- it wouldn't try to analyze the image pixel by pixel. Instead, it would scan the image looking 
for clusters of pixels to see if they formed these important features. 

The same is true with natural language where, in order to understand the meaning of certain words, 
we need to "pay attention" to certain other words. Consider the following text 

  *I went to the park with my dog and threw the ball. It went high in the air.* 

The word *It* in the second sentence is a pronoun and refers to the *the ball* from the previous 
sentence. Pronouns like it, she, they, etc., almost always refer to another noun introduced previously. 
But there are a couple of key words that we need to "pay attention" to in order to resolve that *it* 
refers to *the ball*. Which words are those? 

Consider a slight variation: 

  *I went to the park with my dog and threw the ball. It barked loudly.*

In this case, the first sentence is unchanged, but the change to second sentence now means that 
the *It* in the second sentence refers to *my dog*, not the ball. 

In the first case, to resolve the *It* in the second sentence, the import words are: 

* threw, ball, high, air 

and in the second case, the important words are: 

* dog, barked, loudly 

We can see from this simple example just how challenging the task is. Understanding the meaning of words, 
even in these very simple cases, can involve using words in previous sentences and words that come after 
the word in the current sentence. 

How should we formulate the challenge of attention? The idea is to begin by associating a vector, 
:math:`v_t`, to each element :math:`s_t` in our sequence. For example, to the (partial) input 
sentence *I went to the park*, we would associate five vectors: 

.. math::

    v_{I}, v_{went}, v_{to}, v_{the}, v_{park}

We pass this sequence to the attention network to compute a new sequence of outputs, call them: 

.. math::

    y_{I}, y_{went}, y_{to}, y_{the}, y_{park}

To compute :math:`y_N`, for each *N*, we compute a weighted (normalized) dot product of the 
associated input vector :math:`v_N` with all other vectors: 

.. math:: 

    y_{N} \approx \sum_{t} w_{N,t} ( v_N \cdot v_t )

Intuitively, the dot product is used because it computes a similarity between two vectors.
In the real definition, we also apply an activation function (*softmax*) to convert the raw 
values into a normalized vector that can be interpreted as a probability distribution. 

This is the basic intuition. If you read the original paper, or if you inspect a real-world, 
transformer architecture closely, you will see that in fact each input vector, :math:`v_t`, plays three 
distinct roles in the attention component: that of a *query*, a *key* and a *value*, to perform 
the following computations, respectively:

1. compare it to every other vector to establish the weights for its own output
2. compare it to every other vector to establish the weights for the other outputs
3. use it as part of the weighted sum to compute each output vector once the weights 
   have been established

This is largely a "trick" to enable more efficient computations of the attention matrices. We 
won't go into more details here, but if you are interested, more details can be found in the 
original paper or in a number of online resources. 

.. 
    To motivate the *query*, *key* and *value* notions, we can think of the challenge of 
    determining which features are most important as being similar to search. 
    Suppose we have a giant database of employees, both information about them and an image of them, 
    and a user enters a search query to find a specific employee of interest. We can imagine that, for each 
    employee in the database, we have a set of important information, which we can call "keys" (:math:`k_i`), 
    in the database, things like:
    
    * Name, :math:`k_1`
    * Age, :math:`k_2`
    * Job title, :math:`k_3` 
    * Department, :math:`k_4` 
    * ...

    When a user enters a search query, :math:`q`, what we can do is to try and compute how similar the 
    :math:`q` is to each :math:`k_i`. We define a *similarity metric*, :math:`s(q, k)`, which returns a larger 
    number for objects that are more similar to each other. 
    We then associate the relevant object in the database, in this case, the image, 
    with the value. If we think of :math:`q` and :math:`k` as vectors, we can use the dot product as the 
    similarity metric. 


Language Embedding
^^^^^^^^^^^^^^^^^^
Keep in mind that an ANN cannot work directly on text data. Instead, they require numeric data. Thus, 
we must have a way to translate text into numbers. We can do this is with a *language 
embedding*. 

One way to create an embedding is to write down a list of every possible word that could 
appear and treat each word as a categorical and use one-hot encoding. For example, in English, 
there are nearly 500,000 words with maybe 170,000 or so in current use. Therefore, we could assign 
each word a number between 1 and 500,000, (or 1 and 170,000 if we want to restrict to words in current use)
and we could represent a single word as the array :math:`[0, 0, 0, ..., 0, 1, 0, 0, ... ,0]` with a 1 
in the index of the word. Then, a sequence of words would be represented as a 2d-array, where each 
word in the sequence was represented as a 1d-array. 

Note that this would lead to very sparse data and 
in practice is not a very good approach. Besides being an inefficient representation, this embedding 
produces vectors that all have the same distance from each other. A better embedding would represent 
similar words, such as pizza and pizzas or dog and doggy, with vectors that were a smaller distance 
away from each other. 

The Transformer architecture includes a language encoding component (both for the input to the encoder 
and for the output fed to the decoder) that learns an *embedding 
matrix* with position indexes included in the embedding. In other words, the embedding maps both the 
word *and its position in the sequence* to a numeric value, and these values are improved throughout 
the training process. Essentially, the model learns an embedding of the sparse one-hot encoding
mapping into a much lower-dimensional space. 


Feed-Forward Network 
^^^^^^^^^^^^^^^^^^^^
In addition to the the attention subcomponents, each half of the transformer architecture 
includes a fully connected feed-forward network with 1 hidden layer. These feed-forward networks 
are exactly like the networks we looked at the beginning of Unit 3. In the original paper, 
two convolutions with kernel size 1, input and output dimensionality of 512, and 
inner-layer dimensionality of 2048 were used. 


Transformer Architecture: Why is it successful?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have tried to provide a basic intuition for attention and why it could be important, but what role does the 
attention component play in the greater architecture, and what role, for that matter, does the feed-forward 
component play? The short answer it seems is that no one really knows. 

One intuition that has been given is that the attention mechanism focuses on individual elements of the 
input sequence (individual words, for example), and which elements are important to which other elements. 
The feed-forward network then learns "higher level" patterns --- for example, more complete thoughts or phrases 
in the case of NLP tasks. But to the best of our knowledge, these intuitions cannot rigorously be established.


Transformers: Evolution and Impact Since 2017
----------------------------------------------

The transformer architecture has made great impact since the original 2017 paper. The architecture 
has been applied to many fields and tasks within ML, achieving state-of-the-art performance 
in many cases, including:

* Natural Language Processing (e.g., translation, question and answer, etc.)
* Computer Vision (e.g., object detection, image classification, etc.)
* Audio analysis (e.g., voice/speech recognition, generative music, etc.)
* Multi-modal processing; i.e., multiple types of simultaneous input (e.g., voice and mouse gestures)

In this section 
we survey some of the major advances and how they have been enabled with transformers. 

Encoder-Decoder, Encoder-only and Decoder-only Model Variants 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Recall that when we reviewed the Transformer architecture above, we mentioned that there were 
two halves (a left half and a right half) called the *encoder* and the *decoder*. The difference 
between the two was that the decoder included a *masked* multi-head attention mechanism. The word 
*masked* here refers to the fact that some of the attention matrix for the input sequence is hidden 
from the network. Specifically, the part of the sequence after the index currently being predicted 
is masked. Said differently, with masked attention, positions can only utilize the attention weights 
of positions that precede them. 

Intuitively, we may want to use masking in different ways, or not at all, depending on the task. 
For this reason, encoder-only and decoder-only variants of the transformer model have been created. 

For example, with sentiment analysis, there is no need for masking, as we want the model to be 
able to use the entire input sequence for the prediction. Therefore, we may use an encoder-only 
model for these tasks. 

On the other hand, for the task of text generation or sentence completion (e.g.,autofill), we want 
the model to *only* be able to use the part of the sequence that came before the prediction position. 
Therefore, we may use a decoder-only model for these tasks.  

Finally, for language translation (which was the task originally studied in the 
"Attention Is All You Need" paper), we may want the model to see the entire input language sequence 
but only be able to see the part of the attentions of the words that have already been translated 
in the target language. This gives intuition behind the original encoder-decoder model: the encoder 
utilizes attentions for all of the inputs words (e.g., English), but the decoder can only see the 
attentions of the words that have already been translated (e.g., French).


Model Variations and Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are several important variations that have been explored. 

The first major variant is the number of *layers*. You will notice the *Nx* in the architecture diagram. 
This indicates that the structure is repeated a certain number of times (in the original paper, it was 7).

The *embedding dimension* and *number of attention heads* are also hyperparameters of the transformer, but 
we will not discussed these topics in detail. Also, it seems that in practice, these parameters all 
tend to be scaled together (i.e., increasing the number of layers will lead to increases in the embedding dimension 
and the number of attention heads).

.. figure:: ./images/GPT-3-hyperparams.png
    :width: 700px
    :align: center

    Hyperparameters for different sizes of the GPT-3 model. Taken from the 
    "Language Models are Few-Shot Learners" paper, [4].


There have been attempts to empirically study different aspects of the architecture. One interesting 
paper along these lines is "Training Compute-Optimal Large Language Models", from 2022 [3], sometimes 
referred to as the "Chinchilla paper" after the model they introduce. The paper establishes that current 
models, such as GPT-3, may be undertraining for the model architectures they are using.  



Some Important Transformer Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a quick overview of some of the more important transformer models to be released over the 
last 6 or 7 years: 

* 2017: Attention is all you need paper 

* 2018:

  * GPT (decoder-only): 117M params, 12 layers, 768 emb dim, 12 heads 
  * BERT (BASE) (encoder-only): 110M params, 12 layers, 768 emb dim, 12 heads 

* 2019: 

  * GPT-2 (XL): 1.5B params, 48 layers, 768 emb dim, 25 heads

* 2020: 

  * T5 (11B) (decoder only): 11B params, 24 layers, 1024 emd dim, 128 heads 
  * GPT-3: 175B params, 96 layers, 12288 emb dim, 96 heads

* 2022:

  * Chinchilla: **70B params**, 80 layers, 8192 emb dim, 64 heads. (Notably smaller, as that 
    was the point of the paper)
  * PaLM (decoder-only): 540B params


* 2023:

  * GPT-4: *Details unknown* 


Training Transformers 
^^^^^^^^^^^^^^^^^^^^^

All of the large transformer models (including those listed above) have been trained on a very 
large amount of data. 

They utilize a technique called *self-supervised learning* where the model can use data that has not been 
manually labeled. Examples of this technique include:

1. Taking a large corpus of text and masking random words. For example, the 2019 BERT model was 
   trained on text by masking 15% of all words randomly. 
2. For sequence to sequence tasks (e.g., language translation), encoding the task to perform in the 
   input sequence and masking the output sequence. For example, "Translate the following English to 
   Russian: We threw the ball in the park." This approach requires a corpus of translations. 

And to be clear, these are large input sets. To give a sense, the following lists of the 
large sources of texts that one or more of the above models was trained on: 

* Common Crawl: An open repository of web crawl data maintained by the non-profit of the same name. 
  The Feb/March 2024 crawl contains 3.16 billion pages and is over 90 TB compressed. [5]
* Colossal Clean Crawl Corpus (C4): a filtered/cleaned up version of the Common Crawl 
* WebText: Introduced by OpenAI in the GPT-3 paper [4], it analyzed and scraped outbound Reddit links deemed to 
  be of high quality and then applied some filtering/post-processing (e.g., deduplication) to clean it up. 
  About 8M documents in total, 40GB of text. 
* Wikipedia: About 60M pages, 22GB compressed. 
* GitHub code repositories: details seem to be somewhat unclear as to what exactly has been used. 

From these large collections of text, the model learns the foundations of language, but it will not 
necessarily perform well on specific tasks. For that, we use fine-tuning, also called *transfer learning*.
The idea is to further train the (pre-trained) language model with a much smaller set of human labeled 
data for a specific task. For example, if you were training a model to do question and answer about the 
UT campus while giving tours, you might create a labeled dataset of questions and answers about the usage 
and history of various building on campus. 

While not all the details are known, the computing costs to pre-train these models are likely also very large, 
with some notable exceptions. For instance, some estimate the cost to train GPT-3 to be in the $10Ms. 


Additional References
----------------------

1. Vaswani, et al. "Attention Is All You Need." July, 2017. https://arxiv.org/abs/1706.03762
2. MIT 6.S191: Recurrent Neural Networks, Transformers, and Attention. http://introtodeeplearning.com
3. Hoffman et al. Training Compute-Optimal Large Language Models. March, 2022. https://arxiv.org/abs/2203.15556. 
4. Brown, et al. Language Models are Few-Shot Learners. 2020. https://arxiv.org/pdf/2005.14165.pdf
5. Common Crawl. Feb-March 2024 Data. https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/index.html
6. C4 (Colossal Clean Crawled Corpus). https://paperswithcode.com/dataset/c4



Acknowledgements
-----------------

.. [1] Significant portions of the material in this section were based in part on the excellent MIT lecture, 
       Recurrent Neural Networks, Transformers, and Attention, which is part of the 
       6.S191: Introduction to Deep Learning course. 