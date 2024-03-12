# Chat with your Database ## Introduction

Imagine chatting with your database using just a few lines of code. Sounds unbelievable, right? Well, believe it! We'll show you how you can effortlessly chat with a huge database containing 10 million business records—all with just a few lines of SuperDuperDB code. 

Here is the behemoth 10M dataset: [FREE COMPANY DATASET](https://app.snowflake.com/marketplace/listing/GZSTZRRVYL2/people-data-labs-free-company-dataset)

Here is the [Google Colab Notebook](https://colab.research.google.com/drive/1YXzAuuQdqkWEQKchglxUuAMzNTKLu5rC#scrollTo=0Zf4Unc_fNBp) for you to quickstart!

Chatting with this type of massive dataset using the standard RAG pipeline is next to impossible due to the cost and scale. However, with SuperDuperDB, you can achieve the same functionality with just a few lines of code.

You can control the low level code while enjoying writing the high level code! So that, you can increase the capacity of it! Whether you're using `Snowflake` or `any other SQL dataset`, we've got your back.

Here's the simplicity of it:
1. Connect using your URI (works with any SQL Database).
2. Specify your Database/Table Name.
3. Craft a query in plain English.

You'll not only get results but also clear explanations!

Let SuperDuperDB and OpenAI do the heavy lifting—all within a single prompt. Stay tuned for more exciting features, including prompt chaining!

Bring the power of AI into your database effortlessly! 

Let's bring AI into your database! 🚀
## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:

!pip install superduperdb
!pip install openai numpy==1.24.4
!pip install git+https://github.com/SuperDuperDB/superduperdb.git@66b457eb99f5ad9b88d10bb46f8c2d62ddbc1a1d

### Import SuperDuperDB and connect your database

Here we have connected with a mega database from `Snowflake` but it works with all other `SQL` database. 


```python
from superduperdb import superduper
from superduperdb.backends.ibis.query import RawSQL

user = "superduperuser"
password = "superduperpassword"
account = "XXXX-XXXX"  # ORGANIZATIONID-USERID
database = "FREE_COMPANY_DATASET/PUBLIC"  # DATABASE/SCHEMA 

# Here we are using Snowflake FREE_COMPANY_DATASET with 10 million company data
snowflake_uri = f"snowflake://{user}:{password}@{account}/{database}"

# Let's superduper your database
db = superduper(
    snowflake_uri,
    metadata_store='sqlite:///your_database_name.db', # We need a persistent metadata store to store important infos like job. It could be anything including your own database. Here we are using a SQLite database. You can use your same database as well. In that case you don't have to add metadata_store, we will use the same database as metadata store
)
```

### Add OpenAI API Key

If you don't have any, call Sam Altman!


```python
import os

from superduperdb.ext.openai import OpenAIChatCompletion
from IPython.display import Markdown

# Add your OPEN_AI_API_KEY or keep it in your os.environ, we will pick it up from environment
os.environ['OPENAI_API_KEY'] = 'sk-XXX_SAM_ALTMAN_IS_NOT_FIRED_XXX'
```

### Create a helper function to chat your database. 

Here you can tweak the prompts or you can leave here as it is!


```python
def chat_with_your_database(table_name, query, limit = 5):
    # Define the search parameters
    search_term = f'Write me a SQL query for the table named {table_name}. The query is: {query}'

    # Define the prompt for the OpenAIChatCompletion model
    prompt = (
        'Act as a database administrator, and expert in SQL. You will be helping me write complex SQL queries. I will explain you my needs, you will generate SQL queries against my database. The database is a Snowflake database, please take it into consideration when generating SQL.'
        f' I will provide you with a description of the structure of my tables. You must remember them and use them for generating SQL queries.\n'
        'Here are the tables in CSV format: {context}\n\n'
        f'Generate only the SQL query. Always write "regex_pattern" in every "WHERE" query. Integrate a "LIMIT {limit}" clause into the query. Exclude any text other than the SQL query itself. Do not include markdown "```" or "```sql" at the start or end.'
        'Here\'s the CSV file:\n'
    )

    # Add the OpenAIChatCompletion instance to the database
    db.add(OpenAIChatCompletion(model='gpt-3.5-turbo', prompt=prompt))
    
    # Use the OpenAIChatCompletion model to predict the next query based on the provided context
    output, context = db.predict(
        model_name='gpt-3.5-turbo',
        input=search_term,
        context_select=db.execute(RawSQL(f'DESCRIBE {table_name}')).as_pandas().to_csv()
        # context_select=db.execute(RawSQL(f'SELECT * FROM {table_name} LIMIT 10')).as_pandas().to_csv() # Use in case of some other SQL databases like Postgres where `DESCRIBE` is not supported.
    )
    
    try:
        # Attempt to execute the predicted SQL query and retrieve the result as a pandas DataFrame
        # print(output.content)
        query_result = db.execute(RawSQL(output.content)).as_pandas()
        
        if query_result.empty:
            query_result = "No result found. Please edit your query based on the database. Be specific. Like keep everything in lowercase. Use regex etc. Run the same thing multiple times. Always."
    except:
        # If an exception occurs, provide a message to guide the user on adjusting their query
        query_result = "Please edit your query based on the database so that we can find you a suitable result. Please check your table schema if you encounter issues. Run again, if necessary."

    return query_result
```

### Create another helper function to explain the result

This function will be used to explain the result


```python
def explain_the_result(query_result):
    # Define the search parameters
    try:
        search_term = f'Find business insights from it {query_result.to_csv()}'
    except:
        return "No result found. Run again. Please edit your query. Be specific. And always run again. LLM will catch the error and will show you perfect result in multiple attempts."

    # Define the prompt for the OpenAIChatCompletion model
    prompt = (
        f'Assume the role of a database analyst. Your objective is to provide accurate business insights based on the provided CSV content. Avoid reproducing the same CSV file or rewriting the SQL query. Conclude your response with a summary.\n'
        'Context: {context}'
        'Here\'s the CSV file for you to analyze:\n'
    )
    
    # Add the OpenAIChatCompletion instance to the database
    db.add(OpenAIChatCompletion(model='gpt-3.5-turbo', prompt=prompt))
    
    # Use the OpenAIChatCompletion model to predict insights based on the provided context
    output, context = db.predict(
        model_name='gpt-3.5-turbo',
        input=search_term,
    )
    
    try:
        # Attempt to format the predicted output as Markdown
        query_result = Markdown(output.content)
    except:
        # If an exception occurs, provide a message to guide the user on adjusting their input
        query_result = "Please edit your input based on the dataset so that we can find you a suitable output. Please check your data if you encounter issues."

    return query_result
```

### Now let's start chatting with your database. 

Run this multiple times as it will keep its context. Here you just edit the `table_name` and `query` to see the final result.


```python
# If you see no result, Run this codeblock multiple times to make the gpt-3.5-turbo work better and change your query as well. Idea: start with a simple query. Then make it gradually complex.

table_name = "FREECOMPANYDATASET"
query = "Find me some company in germany in berlin. Keep all in lower case."

result = chat_with_your_database(table_name, query)

result
```

### Let's explain the table

Table doen't make sense to you, let LLM explain the table above


```python
# Call the explain_the_result function to analyze and explain the business insights
## Run multiple times if no result shown
explain_the_result(query_result=result)
```

### Let's generate result on the fly by prompt chaining

Now you can do model-chaining as well, if you only care about the explanations. Here we found from the dataset about the company 


```python
# Run multiple times if no result shown

table_name = "FREECOMPANYDATASET"
query = "Find me information about BMW company in Germany. Keep all in lower case."

# The result is generated from your dataset. Tweak limit params if you want specific results.
explain_the_result(chat_with_your_database(table_name, query, limit=1))
```

## Let's chat realtime. 

### Ask questions, get result.

We just boiled the whole thing in one function.

Rerun this for new questions. Don't worry, it is keeping the context!

Let's have one simple interface. Where you just write your query and see the result. Simple.


```python
# Run multiple times if no result shown
table_name = "FREECOMPANYDATASET"

# Be innovative and specific here 
query = "Find me information about Volkswagen company in Germany. Keep all in lower case."

def integrated_result(table_name, query):
  queried_result = chat_with_your_database(table_name, query)
  explained_result = explain_the_result(queried_result)

  display(queried_result, explained_result)

# Showing the result here
integrated_result(table_name, query)
```

## Voila! You just had a conversation with your database. Let's take it from here.

This is just the beginning – feel free to customize prompts for your dataset. One secret tips: Mentioning your database schema in the `chat_your_database` function enhances accuracy by a few mile. Another one is giving more data to it. Anyway, it's yours. Play with it. The better you prompt, the better result you get. This prompt of us is just a simple one works for everything! Your journey with SuperDuperDB is in your hands now. Let the exploration begin!

#### Give us a star. We will release more update in this example like visualization, fine tuning, prompt chaining etc.
