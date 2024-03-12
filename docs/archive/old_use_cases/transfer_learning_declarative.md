# Transfer Learning with Sentence Transformers and Scikit-Learn

## Introduction

In this notebook, we will explore the process of transfer learning using SuperDuperDB. We will demonstrate how to connect to a MongoDB datastore, load a dataset, create a SuperDuperDB model based on Sentence Transformers, train a downstream model using Scikit-Learn, and apply the trained model to the database. Transfer learning is a powerful technique that can be used in various applications, such as vector search and downstream learning tasks.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install ipython numpy datasets sentence-transformers numpy==1.24.4
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database
db = superduper(mongodb_uri, artifact_store='filesystem://./data/')

# Reference a collection called transfer
collection = Collection('transfer')
```

## Load Dataset

Transfer learning can be applied to any data that can be processed with SuperDuperDB models.
For our example, we will use a labeled textual dataset with sentiment analysis.  We'll load a subset of the IMDb dataset.


```python
import numpy
from datasets import load_dataset
from superduperdb import Document as D

# Load IMDb dataset
data = load_dataset("imdb")

# Set the number of data points for training (adjust as needed)
N_DATAPOINTS = 100

# Prepare training data
train_data = [
    D({'_fold': 'train', **data['train'][int(i)]})
    for i in numpy.random.permutation(len(data['train']))
][:N_DATAPOINTS]

# Prepare validation data
valid_data = [
    D({'_fold': 'valid', **data['test'][int(i)]})
    for i in numpy.random.permutation(len(data['test']))
][:N_DATAPOINTS // 10]

# Insert training data into the 'collection' SuperDuperDB collection
db.execute(collection.insert_many(train_data))
```

## Run Model

We'll create a SuperDuperDB model based on the `sentence_transformers` library. This demonstrates that you don't necessarily need a native SuperDuperDB integration with a model library to leverage its power. We configure the `Model wrapper` to work with the `SentenceTransformer class`. After configuration, we can link the model to a collection and daemonize the model with the `listen=True` keyword.


```python
import sentence_transformers
from sklearn.svm import SVC

from superduperdb import Model
from superduperdb.ext.numpy import array
from superduperdb.ext.sklearn import Estimator
from superduperdb import superduper
from superduperdb.components.stack import Stack


# Create a SuperDuperDB Model using Sentence Transformers
m1 = Model(
    identifier='all-MiniLM-L6-v2',
    object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
    encoder=array('float32', shape=(384,)),
    predict_method='encode',
    batch_predict=True,
    predict_X='text',
    predict_select=collection.find(),
    predict_kwargs={'show_progress_bar': True},
)


# Create a SuperDuperDB model with an SVC classifier
m2 = Estimator(
    'svc',
    object=SVC(gamma='scale', class_weight='balanced', C=100, verbose=True),
    postprocess=lambda x: int(x),
    train_X='_outputs.text.all-MiniLM-L6-v2.0',
    train_y='label',
    train_select=collection.find(),
    predict_X='_outputs.text.all-MiniLM-L6-v2.0',
    predict_select=collection.find({'_fold': 'valid'})
)

stack = Stack('my-stack', components=[m1, m2])
```


```python
db.add(stack)
```

## Verification

To verify that the process has worked, we can sample a few records to inspect the sanity of the predictions.


```python
# Query a random document from the 'collection' SuperDuperDB collection
r = next(db.execute(collection.aggregate([{'$match': {'_fold': 'valid'}},{'$sample': {'size': 1}}])))

# Print a portion of the 'text' field from the random document
print(r['text'][:100])

# Print the prediction made by the SVC model stored in '_outputs'
print(r['_outputs']['text']['svc']['0'])
```
