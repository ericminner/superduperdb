"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[7752],{79686:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>d,contentTitle:()=>o,default:()=>u,frontMatter:()=>s,metadata:()=>r,toc:()=>c});var a=t(85893),i=t(11151);const s={},o="Sentiment Analysis Using transformers on MongoDB",r={id:"use_cases/classical_tasks/sentiment_analysis_use_case",title:"Sentiment Analysis Using transformers on MongoDB",description:"In this document, we're doing sentiment analysis using Hugging Face's transformers library. We demonstrate that you can perform this task seamlessly in SuperDuperDB, using MongoDB to store the data.",source:"@site/content/use_cases/classical_tasks/sentiment_analysis_use_case.md",sourceDirName:"use_cases/classical_tasks",slug:"/use_cases/classical_tasks/sentiment_analysis_use_case",permalink:"/docs/use_cases/classical_tasks/sentiment_analysis_use_case",draft:!1,unlisted:!1,editUrl:"https://github.com/SuperDuperDB/superduperdb/blob/main/docs/hr/content/use_cases/classical_tasks/sentiment_analysis_use_case.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Image Feature-Store Using Torchvision on MongoDB",permalink:"/docs/use_cases/classical_tasks/resnet_features"},next:{title:"Transfer-Learning Using Transformers and Scikit-Learn on MongoDB",permalink:"/docs/use_cases/classical_tasks/transfer_learning"}},d={},c=[{value:"Connect to datastore",id:"connect-to-datastore",level:2}];function l(e){const n={code:"code",h1:"h1",h2:"h2",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(n.h1,{id:"sentiment-analysis-using-transformers-on-mongodb",children:"Sentiment Analysis Using transformers on MongoDB"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"!pip install superduperdb\n!pip install datasets\n"})}),"\n",(0,a.jsxs)(n.p,{children:["In this document, we're doing sentiment analysis using Hugging Face's ",(0,a.jsx)(n.code,{children:"transformers"})," library. We demonstrate that you can perform this task seamlessly in SuperDuperDB, using MongoDB to store the data."]}),"\n",(0,a.jsx)(n.p,{children:"Sentiment analysis has some real-life use cases:"}),"\n",(0,a.jsxs)(n.ol,{children:["\n",(0,a.jsxs)(n.li,{children:["\n",(0,a.jsxs)(n.p,{children:[(0,a.jsx)(n.strong,{children:"Customer Feedback & Review Analysis:"})," Analyzing customer reviews and feedback to understand overall satisfaction, identify areas for improvement, and respond to customer concerns. It is used in the E-commerce industry frequently."]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:["\n",(0,a.jsxs)(n.p,{children:[(0,a.jsx)(n.strong,{children:"Brand Monitoring:"})," Monitoring social media, blogs, news articles, and online forums to gauge public sentiment towards a brand, product, or service. Addressing negative sentiment and capitalizing on positive feedback."]}),"\n"]}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Sentiment analysis plays a crucial role in understanding and responding to opinions and emotions expressed across various domains, contributing to better decision-making and enhanced user experiences."}),"\n",(0,a.jsxs)(n.p,{children:["All of these can be done with your ",(0,a.jsx)(n.code,{children:"existing database"})," and ",(0,a.jsx)(n.code,{children:"SuperDuperDB"}),". You can integrate similar code into your ETL infrastructure as well. Let's see an example."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from datasets import load_dataset\nimport numpy\nfrom transformers import AutoTokenizer, AutoModelForSequenceClassification\n\n# Import Document (aliased as D) and Dataset from the superduperdb module\nfrom superduperdb import Document as D, Dataset\n"})}),"\n",(0,a.jsx)(n.h2,{id:"connect-to-datastore",children:"Connect to datastore"}),"\n",(0,a.jsx)(n.p,{children:'SuperDuperDB can work with MongoDB (one of many supported databases) as its database backend. To make this connection, we\'ll use the Python MongoDB client, pymongo, and "wrap" our database to transform it into a SuperDuper Datalayer.'}),"\n",(0,a.jsxs)(n.p,{children:["First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the ",(0,a.jsx)(n.code,{children:"MongoDB_URI"})," based on your specific setup."]}),"\n",(0,a.jsx)(n.p,{children:"Here are some examples of MongoDB URIs:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:["For testing (default connection): ",(0,a.jsx)(n.code,{children:"mongomock://test"})]}),"\n",(0,a.jsxs)(n.li,{children:["Local MongoDB instance: ",(0,a.jsx)(n.code,{children:"mongodb://localhost:27017"})]}),"\n",(0,a.jsxs)(n.li,{children:["MongoDB with authentication: ",(0,a.jsx)(n.code,{children:"mongodb://superduper:superduper@mongodb:27017/documents"})]}),"\n",(0,a.jsxs)(n.li,{children:["MongoDB Atlas: ",(0,a.jsx)(n.code,{children:"mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"})]}),"\n"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"import os\nfrom superduperdb.backends.mongodb import Collection\nfrom superduperdb import superduper\n\n# Set an environment variable to enable PyTorch MPS fallback\nos.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n\n# Get the MongoDB URI from the environment variable \"MONGODB_URI,\" defaulting to \"mongomock://test\"\nmongodb_uri = os.getenv(\"MONGODB_URI\",\"mongomock://test\")\n\n# SuperDuperDB, now handles your MongoDB database\n# It just super dupers your database \ndb = superduper(mongodb_uri)\n\n# Collection instance named 'imdb' in the database\ncollection = Collection('imdb')\n"})}),"\n",(0,a.jsx)(n.p,{children:"We train the model using the IMDB dataset."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"# Load the IMDb dataset using the load_dataset function from the datasets module\ndata = load_dataset(\"imdb\")\n\n# Set the number of datapoints to be used for training and validation. Increase this number to do serious training\nN_DATAPOINTS = 4\n\n# Insert randomly selected training datapoints into the 'imdb' collection in the database\ndb.execute(collection.insert_many([\n    # Insert training data into your database from the dataset. Create Document instances for each training datapoint, setting '_fold' to 'train'\n    D({'_fold': 'train', **data['train'][int(i)]}) for i in numpy.random.permutation(len(data['train']))[:N_DATAPOINTS]\n]))\n\n# Insert randomly selected validation datapoints into the 'imdb' collection in the database\ndb.execute(collection.insert_many([\n    # Insert validation data into your database from the dataset. Create Document instances for validation datapoint, setting '_fold' to 'valid'\n    D({'_fold': 'valid', **data['test'][int(i)]}) for i in numpy.random.permutation(len(data['test']))[:N_DATAPOINTS]\n]))\n"})}),"\n",(0,a.jsx)(n.p,{children:"Retrieve a sample from the database."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"# Execute the find_one() method to retrieve a single document from the 'imdb' collection. To check if the database insertion is done okay.\nr = db.execute(collection.find_one())\nr\n"})}),"\n",(0,a.jsx)(n.p,{children:"Build a tokenizer and utilize it to create a data collator for batching inputs."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"tokenizer = AutoTokenizer.from_pretrained(\"distilbert-base-uncased\")\n\n# Instantiate a sequence classification model for the 'distilbert-base-uncased' model with 2 labels\nmodel = AutoModelForSequenceClassification.from_pretrained(\"distilbert-base-uncased\", num_labels=2)\n\n# Create a Pipeline instance for sentiment analysis\n# identifier: A unique identifier for the pipeline\n# task: The type of task the pipeline is designed for, in this case, 'text-classification'\n# preprocess: The tokenizer to use for preprocessing\n# object: The model for text classification\n# preprocess_kwargs: Additional keyword arguments for the tokenizer, e.g., truncation\nmodel = Pipeline(\n    identifier='my-sentiment-analysis',\n    task='text-classification',\n    preprocess=tokenizer,\n    object=model,\n    preprocess_kwargs={'truncation': True},\n)\n"})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"# Assuming 'This is another test' is the input text for prediction\n# You're making a prediction using the configured pipeline model\n# with the one=True parameter specifying that you expect a single prediction result.\nmodel.predict('This is another test', one=True)\n"})}),"\n",(0,a.jsx)(n.p,{children:"We'll assess the model using a straightforward accuracy metric. This metric will be recorded in the model's metadata as part of the training process."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"# Import TransformersTrainerConfiguration from the superduperdb.ext.transformers module\nfrom superduperdb.ext.transformers import TransformersTrainerConfiguration\n\n# Create a configuration for training a transformer model\ntraining_args = TransformersTrainerConfiguration(\n    identifier='sentiment-analysis',  # A unique identifier for the training configuration\n    output_dir='sentiment-analysis',  # The directory where model predictions will be saved\n    learning_rate=2e-5,  # The learning rate for training the model\n    per_device_train_batch_size=2,  # Batch size per GPU (or CPU) for training\n    per_device_eval_batch_size=2,  # Batch size per GPU (or CPU) for evaluation\n    num_train_epochs=2,  # The number of training epochs\n    weight_decay=0.01,  # Weight decay for regularization\n    save_strategy=\"epoch\",  # Save model checkpoints after each epoch\n    use_cpu=True,  # Use CPU for training (set to False if you want to use GPU)\n    evaluation_strategy='epoch',  # Evaluate the model after each epoch\n    do_eval=True,  # Perform evaluation during training\n)\n"})}),"\n",(0,a.jsx)(n.p,{children:"Now we're ready to train the model:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"# Import the Metric class from the superduperdb module\nfrom superduperdb import Metric\n\n# Fit the model using training data and specified configuration\nmodel.fit(\n    X='text',  # Input data (text)\n    y='label',  # Target variable (label)\n    db=db,\n\n  # Super Duper wrapped Database connection\n    select=collection.find(),  # Specify the data to be used for training (fetch all data from the collection)\n    configuration=training_args,  # Training configuration using the previously defined TransformersTrainerConfiguration\n    validation_sets=[\n        # Define a validation dataset using a subset of data with '_fold' equal to 'valid'\n        Dataset(\n            identifier='my-eval',\n            select=collection.find({'_fold': 'valid'}),\n        )\n    ],\n    data_prefetch=False,  # Disable data prefetching during training\n    metrics=[\n        # Define a custom accuracy metric for evaluation during training\n        Metric(\n            identifier='acc',\n            object=lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x)\n        )\n    ]\n)\n"})}),"\n",(0,a.jsx)(n.p,{children:"We can confirm that the model produces sensible predictions by examining the output. If you are okay with the performance, you may predict it on your whole database and save it for future reference. All can be done on SuperDuperDB in real-time."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'# Assuming "This movie sucks!" is the input text for sentiment analysis\n# You\'re making a prediction using the configured pipeline model\n# with the one=True parameter specifying that you expect a single prediction result.\nsentiment_prediction = model.predict("This movie sucks!", one=True)\n\nprint("Sentiment Prediction", sentiment_prediction)\n'})})]})}function u(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(l,{...e})}):l(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>o});var a=t(67294);const i={},s=a.createContext(i);function o(e){const n=a.useContext(s);return a.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:o(e.components),a.createElement(s.Provider,{value:n},e.children)}}}]);