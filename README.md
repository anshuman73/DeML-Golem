# DeML-Golem
Proof Of Concept of DEcentralised Machine Learning on top of the Golem (https://golem.network/) architecture


## The Idea

The basic idea of DeML (Decentralised ML) is to provide a framework for working with Machine Learning models across a network of computers with ease and low computing costs. DeML uses the concepts laid down by Federated Learning to combine the sub-step models it trains on different provider nodes into a full fleged model that can be compared to a model trained completely locally. FL models do definitely suffer from slightly sub-par accuracies, but do not require a single expensive machine to do their computation.


You can learn about FL [here](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) and [here](https://arxiv.org/pdf/1602.05629) or **enjoy a quirky comic [here](https://federated.withgoogle.com/).** FL is definitely more about privacy based systems, and we don't explore that here as much as the foundation it lays for distributed working.


Currently, for the hackathon, MNIST (The "Hello World" of the ML realm) has been used as the proof of concept to showcase this MVP, but as Golem reduces the restrictions [Explained below](#current-limitations) as we move to the mainnet, you can expect to train increasingly difficult and useful models on Golem. [If you're looking at this repository beyond the submission date, chances are there's another branch on the repo with a more complicated model.]


Ideally, I wanted to build a twin-component product as a part of **DeML**, one where I could upload a model and the data file and recieve the inferences from the providers, and one where I can train, but decided to build only the latter due to time constraints and my inexperience with Golem. An orchestrator for producing results from a trained model should be much easier to build. (And useful for models like GANs, which use a lot of computing power.)


## Motivation

The motivation for this project actually comes from a real life incident, where in order to train an ML model I was pursuing my research on, I accidently racked up an AWS bill large enough to buy groceries for a month. As a student, obviously this procedure isn't scalable, and I realised there needs to be a better way. Most of our personal devices are hardly ever enough to run long and complex models.

Free services like Google Colab or FloydHub are great, but with come with their own set of restrictions that make it extremely hard to run variations of models a researcher might need to.

Hence was born the idea of DeML, a way to train your models on nodes provided by the Golem Network by theoritically just writing two components - your dataloader (both locally and on provider) and your ML model!


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

The current set of limitations mostly come from the design of Golem for now. The ones that impact us the most are - 

1. The executor time limit. Currently every executor instance gets a maximum of 30 minutes to run, which means training complex models and leaving them running for a few hours/days is not currently possible.

2. Lack of access of internet connectivity on the nodes. This means that the data to be used for training has to be uploaded (or embedded in the docker image). Both of these have their own set of issues - uploading the dataset to your provider is next to impossible with inter-node communication speeds, and the ones sent along with the docker image (which has a 1Gib limit) might have a bug that force loads them onto the RAM, restricting their size to the ram size (see [this chat thread](https://discord.com/channels/684703559954333727/756161015493951600/795981964418875402) for reference)

Most of these issues will be ironed out in the upcoming weeks, and hopefully that will allow for a more robust usage of such an application.

**The idea is that instead of a high-compute server costing you a bazzilion dollars, you get to train your model on a network of smaller computers, with extremely negligible costs (which is free for now in the rinkby stage!), and still approach accuracies shown by a sequentially trained model**

You can even do [something suggested](https://discord.com/channels/684703559954333727/740956182180528239/796025383317012570) by a community member!

## Innovations + Future Work

### Cool things this Demo does:

    1. Runs a completely customisable model in a distributed way. You can control the number of providers, the epochs on each provider and even how they preprocess the data!
    2. Get logs inidicating the performance of each node in the intermediate steps.
    3. Combine and get your prepared models in different stages!

### What more I would like it seeing doing:

    1. Build a UI. Something like slate, another submission in the hackathon, where I can simply provide my model definition and dataset, and never interact much with the code unless I want to.
    2. Another module to run ML models after we've trained them to process live data - with a UI as well, where I just upload my .h5 files and specify dataset.
    3. Run more complex models!! (Keep an eye out for another branch!)
    4. Run ML models on GPU! The golem team is working on bringing in GPU support, and it will be amazing to see the performance bump that could possibly bring to ML training, especially for models like CNNs etc
