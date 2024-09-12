#!/usr/bin/env python
# coding: utf-8

# # Building and Debugging a Custom LangChain Chatbot with External Information Integration

# ## Key take aways
# * Create a chatbot using LangChain’s Chat Model
# * Use Prompt Template, and Output Parser to organize the output and see only the text
# * Learn how is chain implement in `LangChain`
# * Integrate external information (RAG) to guide model responses
# * Implement streaming to get the output being generated and ready
# * Finally, debug the chain

# # Installation of necessary libraries
# In this tutorial, we will be using the `claude-3-5-sonnet-20240620` model developed by Anthropic. Let's install the necessary libiraries, including:
# * `langchain`
# * `langchain_core`
# * `lanchain_anthropic`
# 
# For this purpose we will be using the `pip` command, as follows: 

# In[ ]:


# Install the essential libraries
get_ipython().system(' pip install langchain langchain_core langchain_anthropic')


# # Setting the API key
# 
# After the successfull install of the required libraries, we would be required to using the API key for the Antrhopic model. To get an API key you can visit visit "https://console.anthropic.com". 
# 
# To set the API key, i.e., 'ANTHROPIC_API_KEY', we will be using the `os` module. The Python's built-in `os` module allows interaction with the operating system, including environment variables, file paths, and system commands. The `os.environ` is a dictionary-like object that stores environment variables. This object will be used to sets an environment variable called `ANTHROPIC_API_KEY` which will store your API key. Environment variables are used to store configuration data (e.g., API keys, secrets) separately from code.
# 
# Let's dive in!

# In[34]:


import os
os.environ['ANTHROPIC_API_KEY']=""
api_key= os.getenv('ANTHROPIC_API_KEY')


# # Import and set the model
# We are set to import the Anthropic's chat model, i.e., `claude-3-5-sonnet-20240620`, as given below:

# In[23]:


from langchain_anthropic import ChatAnthropic
llm= ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.8, max_tokens=250)


# In the following cell, we are using two methods `StrOutputParser` and `ChatPromptTemplate`. The `StrOutputParser` is used to parse the output of the model and access only the string. The `ChatPromptTemplate` is a template designed to structure and format prompts for chat-based LLMs. It allows users to define a conversational context by organizing multiple messages, including system instructions, user inputs, and previous responses.

# In[24]:


#Let's test it!
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template= ChatPromptTemplate.from_messages([("system", "You are an assistant helping users with their queries."), 
                          ("human",  "{query}")])

llm2 = template | llm | StrOutputParser() # Chaning the prompt template, model and output parser

llm2.invoke({"query":"What is your name?"})


# In[25]:


# Let's install the datasets, if not already. 
#! pip install --upgrade pip
#! pip install datasets


# # Implementing RAG with Claude model
# The Retrieval Augmented Generation (RAG), is a technique in which external (private) data is provided to the LLM for analysis and answering relevant user queries to the data. Following is a simple example of the RAG, in which we provide the IMDB data to the model and ask different questions that can only be answered from the provided data. Let's see how it works:

# In[25]:


# For this example, we will consider the IMDB dataset
from datasets import load_dataset
dataset = load_dataset('imdb')


# In[26]:


train_data = dataset['train']
print(train_data[0])


# In[27]:


# Let's create our on RAG_Prompt for the IMDB dataset, loaded in the previos cells.
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Use the given data when responding: {data}.'),
    ("human", "{query}")
])

rag_chain = rag_prompt | llm | StrOutputParser()


# In[28]:


rag_chain.invoke({
    "query": "Can you classify this review as positive or negative? \"The movie was absolutely fantastic with great performances by the cast,\" the LLM can classify it as positive",
    "data": train_data
})


# In[29]:


rag_chain.invoke({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
})


# # Streaming responses of the model
# With the help of streaming we can get parts of the response as they are generated, which allows us to display them resulting in a quircker feedback from the model.
# 
# ## Syntax of using `stream()` method
# 
# For streaming the response we just need to replace the `invoke()` method with the `stream()` method. However, we need to use `for` loop to iterate thought the generated tokens. Inside the `for` loop, we should print the tokens, as given in the following cell: 

# In[30]:


for chunk in rag_chain.stream({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
}):
    print(chunk, end="")


# # Debugging the internals
# 
# In order to monitor and debug the internal of our chain, we need to import the `set_debug` method from the `langchain.globals` liberary. Please note that for this purpose you need to install `langchain` using the command `!pip install langchain`, if already not installed. 
# 
# Use LangChain’s debugging tools to monitor and debug the internals of your chain.
#     * Inspect how the data flows through each component and identify potential bottlenecks or errors in the chain.
# 

# In[32]:


from langchain.globals import set_debug
set_debug(True)

for chunk in rag_chain.stream({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
}):
    print(chunk, end="")

