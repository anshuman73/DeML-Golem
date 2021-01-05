# DeML-Golem
Proof Of Concept of DEcentralised Machine Learning on top of the Golem (https://golem.network/) architecture


## Idea

The basic idea of DeML (Decentralised ML) is to provide a framework for working with Machine Learning models across a network of computers with ease and low computing costs. DeML uses the concepts laid down by Federated Learning to combine the sub-step models it trains on different provider nodes into a full fleged model that can be compared to a model trained completely locally. FL models do definitely suffer from slightly sub-par accuracies, but do not require a single expensive machine to do their computation.


You can learn about FL [here](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) and [here](https://arxiv.org/pdf/1602.05629) or **enjoy a quirky comic [here](https://federated.withgoogle.com/).** FL is definitely more about privacy based systems, and we don't explore that here as much as the foundation it lays for distributed working.


Currently, for the hackathon, MNIST (The "Hello World" of the ML realm) has been used as the proof of concept to showcase this MVP, but as Golem reduces the restrictions (Explained below)[#current-limitations] as we move to the mainnet, you can expect to train increasingly difficult and useful models on Golem. [If you're looking at this repository beyond the submission date, chances are there's another branch on the repo with a more complicated model.]


Ideally, I wanted to build a twin-component product as a part of **DeML**, one where I could upload a model and the data file and recieve the inferences from the providers, and one where I can train, but decided to build only the latter due to time contrainsts and my inexperience with Golem. An orchestrator for producing results from a trained model should be much easier to build. (And useful for models like GANs, which use a lot of computing power.)


## Motivation

The motivation for this project actually comes from a real life incident, where in order to train an ML model I was pursuing my research on, I accidently racked up an AWS bill large enough to buy groceries for a month. As a student, obviously this procedure isn't scalable, and I realised there needs to be a better way.

Free services like Google Colab or FloydHub are great, but with come with their own set of restrictions that make it extremely hard to run variations of models a reseracher might need to.

Hence was born the idea of DeML, a way to train your models on nodes provided by the Golem Network by theoritically just writing two components - your dataloader (both locally and on provider) and your ML model!
(Not to mention the low cost of the computation, which is free for now in the rinkby stage!)


## Instructions to run locally

The current implementation is extremely easy to start with, and you can simply get started by editing out the ```model_base.py``` with your custom model and try out if you'd like a more complex model on the same dataset.

Here are the proper steps to get started - 

### Step 1
 Install the required dependancies mentioned in the Pipfile. You need to have ```pipenv``` installed. You can do that by ```pip install pipenv```. This tool allows you to create an environment and install the dependancies directly using ```pipenv install``` (after creating an env with ```pipenv --three```)

### Step 2
 Once you have the environment set up, all you need to do is confirm you have a working tensorflow instance (some machines cannot compile TF) 
 You can do something like this inside the env
 (use ```pipenv shell``` to open a shell in the environment)
 ```
 >>> import tensorflow as tf
 >>> hello = tf.constant("hello TensorFlow!")
 >>> sess=tf.Session() 
 >>> print sess.run(hello)
 ```
 If this works, you're set to go!

### Step 3
 Run the orchestrator! Simply do
 ```python provider_orchestrator.py``` to start up your training!

## Current Limitations


## Innovations + Future Work

