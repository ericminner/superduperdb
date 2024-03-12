# Building Private Q&A Assistant Using Mongo and Open Source Model

## Introduction

This notebook is designed to demonstrate how to implement a document Question-and-Answer (Q&A) task using SuperDuperDB in conjunction with open-source model and MongoDB. It provides a step-by-step guide and explanation of each component involved in the process.

Implementing a document Question-and-Answer (Q&A) system using SuperDuperDB, open-source model, and MongoDB can find applications in various real-life scenarios:

1. **Customer Support Chatbots:** Enable a chatbot to answer customer queries by extracting information from documents, manuals, or knowledge bases stored in MongoDB or any other SuperDuperDB supported database using Q&A.

2. **Legal Document Analysis:** Facilitate legal professionals in quickly extracting relevant information from legal documents, statutes, and case laws, improving efficiency in legal research.

3. **Medical Data Retrieval:** Assist healthcare professionals in obtaining specific information from medical documents, research papers, and patient records for quick reference during diagnosis and treatment.

4. **Educational Content Assistance:** Enhance educational platforms by enabling students to ask questions related to course materials stored in a MongoDB database, providing instant and accurate responses.

5. **Technical Documentation Search:** Support software developers and IT professionals in quickly finding solutions to technical problems by querying documentation and code snippets stored in MongoDB or any other database supported by SuperDuperDB. We did that!

6. **HR Document Queries:** Simplify HR processes by allowing employees to ask questions about company policies, benefits, and procedures, with answers extracted from HR documents stored in MongoDB or any other database supported by SuperDuperDB.

7. **Research Paper Summarization:** Enable researchers to pose questions about specific topics, automatically extracting relevant information from a MongoDB repository of research papers to generate concise summaries.

8. **News Article Information Retrieval:** Empower users to inquire about specific details or background information from a database of news articles stored in MongoDB or any other database supported by SuperDuperDB, enhancing their understanding of current events.

9. **Product Information Queries:** Improve e-commerce platforms by allowing users to ask questions about product specifications, reviews, and usage instructions stored in a MongoDB database.

By implementing a document Q&A system with SuperDuperDB, open-source model, and MongoDB, these use cases demonstrate the versatility and practicality of such a solution across different industries and domains.

All is possible without zero friction with SuperDuperDB. Now back into the notebook.

## Prerequisites

Before starting the implementation, make sure you have the required libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install vllm
!pip install sentence_transformers numpy==1.24.4
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

collection = Collection('questiondocs')
```

## Load Dataset

In this example, we use the internal textual data from the `superduperdb` project's API documentation. The objective is to create a chatbot that can offer information about the project. You can either load the data from your local project or use the provided data.

If you have the SuperDuperDB project locally and want to load the latest version of the API, uncomment the following cell:


```python
import glob
import re

ROOT = '../docs/hr/content/docs/'

STRIDE = 3       # stride in numbers of lines
WINDOW = 25       # length of window in numbers of lines

files = sorted(glob.glob(f'{ROOT}/**/*.md', recursive=True))

def get_chunk_link(chunk, file_name):
    # Get the original link of the chunk
    file_link = file_name[:-3].replace(ROOT, 'https://docs.superduperdb.com/docs/docs/')
    # If the chunk has subtitles, the link to the first subtitle will be used first.
    first_title = (re.findall(r'(^|\n)## (.*?)\n', chunk) or [(None, None)])[0][1]
    if first_title:
        # Convert subtitles and splice URLs
        first_title = first_title.lower()
        first_title = re.sub(r'[^a-zA-Z0-9]', '-', first_title)
        file_link = file_link + '#' + first_title
    return file_link

def create_chunk_and_links(file, file_prefix=ROOT):
    with open(file, 'r') as f:
        lines = f.readlines()
    if len(lines) > WINDOW:
        chunks = ['\n'.join(lines[i: i + WINDOW]) for i in range(0, len(lines), STRIDE)]
    else:
        chunks = ['\n'.join(lines)]
    return [{'txt': chunk, 'link': get_chunk_link(chunk, file)}  for chunk in chunks]


all_chunks_and_links = sum([create_chunk_and_links(file) for file in files], [])
```

Otherwise, you can load the data from an external source. The text chunks include code snippets and explanations, which will be utilized to construct the document Q&A chatbot.


```python
# Use !curl to download the 'superduperdb_docs.json' file
!curl -O https://datas-public.s3.amazonaws.com/superduperdb_docs.json

import json
from IPython.display import Markdown

# Open the downloaded JSON file and load its contents into the 'chunks' variable
with open('superduperdb_docs.json') as f:
    all_chunks_and_links = json.load(f)
```

View the chunk content:


```python
from IPython.display import *

# Assuming 'chunks' is a list or iterable containing markdown content
chunk_and_link = all_chunks_and_links[48]
print(chunk_and_link['link'])
Markdown(chunk_and_link['txt'])
```

The chunks of text contain both code snippets and explanations, making them valuable for constructing a document Q&A chatbot. The combination of code and explanations enables the chatbot to provide comprehensive and context-aware responses to user queries.

As usual we insert the data. The `Document` wrapper allows `superduperdb` to handle records with special data types such as images,
video, and custom data-types.


```python
from superduperdb import Document

# Insert multiple documents into the collection
insert_ids = db.execute(collection.insert_many([Document(chunk_and_link) for chunk_and_link in all_chunks_and_links]))
print(insert_ids[:5])
```

## Create a Vector-Search Index

To enable question-answering over your documents, set up a standard `superduperdb` vector-search index using `sentence_transformers` (other options include `torch`, `openai`, `transformers`, etc.).

A `Model` is a wrapper around a self-built or ecosystem model, such as `torch`, `transformers`, `openai`.


```python
import sentence_transformers
from superduperdb import Model, vector

model = Model(
    identifier='embedding', 
    object=sentence_transformers.SentenceTransformer('BAAI/bge-large-en-v1.5'),
    encoder=vector(shape=(1024,)),
    predict_method='encode', # Specify the prediction method
    postprocess=lambda x: x.tolist(),  # Define postprocessing function
    batch_predict=True, # Generate predictions for a set of observations all at once 
)
```


```python
vector = model.predict('This is a test', one=True)
print('vector size: ', len(vector))
```

A `Listener` essentially deploys a `Model` to "listen" to incoming data, computes outputs, and then saves the results in the database via `db`.


```python
# Import the Listener class from the superduperdb module
from superduperdb import Listener


# Create a Listener instance with the specified model, key, and selection criteria
listener = Listener(
    model=model,          # The model to be used for listening
    key='txt',            # The key field in the documents to be processed by the model
    select=collection.find()  # The selection criteria for the documents
)
```

A `VectorIndex` wraps a `Listener`, allowing its outputs to be searchable.


```python
# Import the VectorIndex class from the superduperdb module
from superduperdb import VectorIndex

# Add a VectorIndex to the SuperDuperDB database with the specified identifier and indexing listener
_ = db.add(
    VectorIndex(
        identifier='my-index',        # Unique identifier for the VectorIndex
        indexing_listener=listener    # Listener to be used for indexing documents
    )
)
```


```python
# Execute a find_one operation on the SuperDuperDB collection
document = db.execute(collection.find_one())
document.content['txt']
```


```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D
from IPython.display import *

# Define the query for the search
# query = 'Code snippet how to create a `VectorIndex` with a torchvision model'
query = 'can you explain vector-indexes with `superduperdb`?'

# Execute a search using SuperDuperDB to find documents containing the specified query
result = db.execute(
    collection
        .like(D({'txt': query}), vector_index='my-index', n=5)
        .find()
)

# Display a horizontal rule to separate results
display(Markdown('---'))

# Display each document's 'txt' field and separate them with a horizontal rule
for r in result:
    display(Markdown(r['txt']))
    display(r['link'])
    display(Markdown('---'))
```

## Create a LLM Component

In this step, a LLM component is created and added to the system. This component is essential for the Q&A functionality:


```python
from superduperdb.ext.llm.vllm import VllmModel

# Define the prompt for the llm model
prompt_template = (
    'Use the following description and code snippets about SuperDuperDB to answer this question about SuperDuperDB\n'
    'Do not use any other information you might have learned about other python packages\n'
    'Only base your answer on the code snippets retrieved and provide a very concise answer\n'
    '{context}\n\n'
    'Here\'s the question:{input}\n'
    'answer:'
)

# Create an instance of llm with the specified model and prompt
llm = VllmModel(identifier='llm',
                 model_name='mistralai/Mistral-7B-Instruct-v0.2', 
                 prompt_template=prompt_template,
                 inference_kwargs={"max_tokens":512})

# Add the llm instance
db.add(llm)

# Print information about the models in the SuperDuperDB database
print(db.show('model'))
```

## Ask Questions to Your Docs

Finally, you can ask questions about the documents. You can target specific queries and use the power of MongoDB for vector-search and filtering rules. Here's an example of asking a question:


```python
from superduperdb import Document
from IPython.display import Markdown

def question_the_doc(question):
    # Use the SuperDuperDB model to generate a response based on the search term and context
    output, sources = db.predict(
        model_name='llm',
        input=question,
        context_select=(
            collection
                .like(Document({'txt': question}), vector_index='my-index', n=5)
                .find()
        ),
        context_key='txt',
    )
    
    # Get the reference links corresponding to the answer context
    links = '\n'.join(sorted(set([source.unpack()['link'] for source in sources])))
    
    # Display the generated response using Markdown
    return Markdown(output.content + f'\n\nrefs: \n\n{links}')
```


```python
question_the_doc("can you explain vector-indexes with `superduperdb`?'")
```


```python
question_the_doc("What databases and AI frameworks does SuperDuperDB support?")
```

## Now you can build an API as well just like we did
### FastAPI Question the Docs Apps Tutorial
This tutorial will guide you through setting up a basic FastAPI application for handling questions with documentation. The tutorial covers both local development and deployment to the Fly.io platform.
https://github.com/SuperDuperDB/chat-with-your-docs-backend
