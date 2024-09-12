# Building and Debugging a Custom LangChain Chatbot with External Information Integration

## Key take aways
* Create a chatbot using LangChain’s Chat Model
* Use Prompt Template, and Output Parser to organize the output and see only the text
* Learn how is chain implement in `LangChain`
* Integrate external information (RAG) to guide model responses
* Implement streaming to get the output being generated and ready
* Finally, debug the chain

# Installation of necessary libraries
In this tutorial, we will be using the `claude-3-5-sonnet-20240620` model developed by Anthropic. Let's install the necessary libiraries, including:
* `langchain`
* `langchain_core`
* `lanchain_anthropic`

For this purpose we will be using the `pip` command, as follows: 


```python
# Install the essential libraries
! pip install langchain langchain_core langchain_anthropic
```

# Setting the API key

After the successfull install of the required libraries, we would be required to using the API key for the Antrhopic model. To get an API key you can visit visit "https://console.anthropic.com". 

To set the API key, i.e., 'ANTHROPIC_API_KEY', we will be using the `os` module. The Python's built-in `os` module allows interaction with the operating system, including environment variables, file paths, and system commands. The `os.environ` is a dictionary-like object that stores environment variables. This object will be used to sets an environment variable called `ANTHROPIC_API_KEY` which will store your API key. Environment variables are used to store configuration data (e.g., API keys, secrets) separately from code.

Let's dive in!


```python
import os
os.environ['ANTHROPIC_API_KEY']=""
api_key= os.getenv('ANTHROPIC_API_KEY')

```

# Import and set the model
We are set to import the Anthropic's chat model, i.e., `claude-3-5-sonnet-20240620`, as given below:


```python
from langchain_anthropic import ChatAnthropic
llm= ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.8, max_tokens=250)
```

In the following cell, we are using two methods `StrOutputParser` and `ChatPromptTemplate`. The `StrOutputParser` is used to parse the output of the model and access only the string. The `ChatPromptTemplate` is a template designed to structure and format prompts for chat-based LLMs. It allows users to define a conversational context by organizing multiple messages, including system instructions, user inputs, and previous responses.


```python
#Let's test it!
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template= ChatPromptTemplate.from_messages([("system", "You are an assistant helping users with their queries."), 
                          ("human",  "{query}")])

llm2 = template | llm | StrOutputParser() # Chaning the prompt template, model and output parser

llm2.invoke({"query":"What is your name?"})
```




    "My name is Claude. It's nice to meet you!"




```python
# Let's install the datasets, if not already. 
#! pip install --upgrade pip
#! pip install datasets

```

    Requirement already satisfied: pip in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (22.2.2)
    Collecting pip
      Downloading pip-24.2-py3-none-any.whl (1.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m358.6 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 22.2.2
        Uninstalling pip-22.2.2:
          Successfully uninstalled pip-22.2.2
    Successfully installed pip-24.2
    Requirement already satisfied: datasets in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (3.0.0)
    Requirement already satisfied: filelock in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (3.6.0)
    Requirement already satisfied: numpy>=1.17 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (1.21.5)
    Requirement already satisfied: pyarrow>=15.0.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (17.0.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (1.4.4)
    Requirement already satisfied: requests>=2.32.2 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (4.66.5)
    Requirement already satisfied: xxhash in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (3.5.0)
    Requirement already satisfied: multiprocess in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec<=2024.6.1,>=2023.1.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from fsspec[http]<=2024.6.1,>=2023.1.0->datasets) (2024.6.1)
    Requirement already satisfied: aiohttp in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (3.10.5)
    Requirement already satisfied: huggingface-hub>=0.22.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (0.24.6)
    Requirement already satisfied: packaging in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from datasets) (6.0)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (2.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (21.4.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (6.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (1.11.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from huggingface-hub>=0.22.0->datasets) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from requests>=2.32.2->datasets) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from requests>=2.32.2->datasets) (3.3)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from requests>=2.32.2->datasets) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from requests>=2.32.2->datasets) (2024.2.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from pandas->datasets) (2022.1)
    Requirement already satisfied: six>=1.5 in /Users/Bismillah/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)


# Implementing RAG with Claude model
The Retrieval Augmented Generation (RAG), is a technique in which external (private) data is provided to the LLM for analysis and answering relevant user queries to the data. Following is a simple example of the RAG, in which we provide the IMDB data to the model and ask different questions that can only be answered from the provided data. Let's see how it works:


```python
# For this example, we will consider the IMDB dataset
from datasets import load_dataset
dataset = load_dataset('imdb')
```


```python
train_data = dataset['train']
print(train_data[0])
```

    {'text': 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.', 'label': 0}



```python
# Let's create our on RAG_Prompt for the IMDB dataset, loaded in the previos cells.
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Use the given data when responding: {data}.'),
    ("human", "{query}")
])

rag_chain = rag_prompt | llm | StrOutputParser()
```


```python
rag_chain.invoke({
    "query": "Can you classify this review as positive or negative? \"The movie was absolutely fantastic with great performances by the cast,\" the LLM can classify it as positive",
    "data": train_data
})
```




    'Based on the review text "The movie was absolutely fantastic with great performances by the cast," I would classify this as a positive review.\n\nThe language used is overwhelmingly positive, with phrases like:\n- "absolutely fantastic" \n- "great performances"\n\nThese express strong approval and enthusiasm for the movie. There are no negative elements mentioned. The tone is clearly one of praise and recommendation.\n\nSo this review would definitely be classified as positive.'




```python
rag_chain.invoke({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
})
```




    'Based on the review excerpt provided, I can identify the following main positive and negative aspects:\n\nPositive:\n- Cinematography: The review states that "The cinematography was stunning", indicating that the visual aspects of the film were very impressive and well-executed.\n\nNegative:\n- Storyline: The review mentions that "the storyline was weak", suggesting that the plot or narrative of the film was not strong or compelling.\n\nThis short review snippet contrasts the visual strengths of the film with its narrative weaknesses, providing a balanced but critical perspective on the movie.'



# Streaming responses of the model
With the help of streaming we can get parts of the response as they are generated, which allows us to display them resulting in a quircker feedback from the model.

## Syntax of using `stream()` method

For streaming the response we just need to replace the `invoke()` method with the `stream()` method. However, we need to use `for` loop to iterate thought the generated tokens. Inside the `for` loop, we should print the tokens, as given in the following cell: 


```python
for chunk in rag_chain.stream({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
}):
    print(chunk, end="")
```

    Based on the given review snippet, the main positive and negative aspects mentioned are:
    
    Positive:
    - Cinematography: The review states that "The cinematography was stunning", indicating that the visual aspects of the film were very impressive and well-executed.
    
    Negative:
    - Storyline: The review describes the storyline as "weak", suggesting that the plot or narrative of the film was not compelling or well-developed.
    
    This brief review highlights a contrast between the film's strong visual elements and its lacking narrative quality.

# Debugging the internals

In order to monitor and debug the internal of our chain, we need to import the `set_debug` method from the `langchain.globals` liberary. Please note that for this purpose you need to install `langchain` using the command `!pip install langchain`, if already not installed. 

Use LangChain’s debugging tools to monitor and debug the internals of your chain.
    * Inspect how the data flows through each component and identify potential bottlenecks or errors in the chain.



```python
from langchain.globals import set_debug
set_debug(True)

for chunk in rag_chain.stream({
    "query": "What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\" ",
    "data": train_data
}):
    print(chunk, end="")

```

    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence] Entering Chain run with input:
    [0m{
      "input": ""
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > prompt:ChatPromptTemplate] Entering Prompt run with input:
    [0m[inputs]
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > prompt:ChatPromptTemplate] [1ms] Exiting Prompt run with output:
    [0m[outputs]
    [32;1m[1;3m[llm/start][0m [1m[chain:RunnableSequence > llm:ChatAnthropic] Entering LLM run with input:
    [0m{
      "prompts": [
        "System: Use the given data when responding: Dataset({\n    features: ['text', 'label'],\n    num_rows: 25000\n}).\nHuman: What are the main positive and negative aspects mentioned in this review?\n ...\"The cinematography was stunning, but the storyline was weak\""
      ]
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > parser:StrOutputParser] Entering Parser run with input:
    [0m{
      "input": ""
    }
    Based on the given review snippet, I can identify the following main positive and negative aspects:
    
    Positive aspect:
    - Cinematography: The review describes it as "stunning", indicating that the visual aspects of the film were very impressive.
    
    Negative aspect:
    - Storyline: The review states that it was "weak", suggesting that the plot or narrative of the film was not well-developed or engaging.
    
    This brief review contrasts the strong visual elements with the poor storytelling, implying that while the film may have been visually appealing, it lacked in narrative quality.[36;1m[1;3m[llm/end][0m [1m[chain:RunnableSequence > llm:ChatAnthropic] [2.63s] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "Based on the given review snippet, I can identify the following main positive and negative aspects:\n\nPositive aspect:\n- Cinematography: The review describes it as \"stunning\", indicating that the visual aspects of the film were very impressive.\n\nNegative aspect:\n- Storyline: The review states that it was \"weak\", suggesting that the plot or narrative of the film was not well-developed or engaging.\n\nThis brief review contrasts the strong visual elements with the poor storytelling, implying that while the film may have been visually appealing, it lacked in narrative quality.",
            "generation_info": null,
            "type": "ChatGenerationChunk",
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessageChunk"
              ],
              "kwargs": {
                "content": "Based on the given review snippet, I can identify the following main positive and negative aspects:\n\nPositive aspect:\n- Cinematography: The review describes it as \"stunning\", indicating that the visual aspects of the film were very impressive.\n\nNegative aspect:\n- Storyline: The review states that it was \"weak\", suggesting that the plot or narrative of the film was not well-developed or engaging.\n\nThis brief review contrasts the strong visual elements with the poor storytelling, implying that while the film may have been visually appealing, it lacked in narrative quality.",
                "response_metadata": {
                  "stop_reason": "end_turn",
                  "stop_sequence": null
                },
                "type": "AIMessageChunk",
                "id": "run-4bf3fb12-2f23-47c3-a52a-87017dad7116",
                "usage_metadata": {
                  "input_tokens": 69,
                  "output_tokens": 126,
                  "total_tokens": 195
                },
                "tool_calls": [],
                "invalid_tool_calls": []
              }
            }
          }
        ]
      ],
      "llm_output": null,
      "run": null
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > parser:StrOutputParser] [1.62s] Exiting Parser run with output:
    [0m{
      "output": "Based on the given review snippet, I can identify the following main positive and negative aspects:\n\nPositive aspect:\n- Cinematography: The review describes it as \"stunning\", indicating that the visual aspects of the film were very impressive.\n\nNegative aspect:\n- Storyline: The review states that it was \"weak\", suggesting that the plot or narrative of the film was not well-developed or engaging.\n\nThis brief review contrasts the strong visual elements with the poor storytelling, implying that while the film may have been visually appealing, it lacked in narrative quality."
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence] [2.65s] Exiting Chain run with output:
    [0m{
      "output": "Based on the given review snippet, I can identify the following main positive and negative aspects:\n\nPositive aspect:\n- Cinematography: The review describes it as \"stunning\", indicating that the visual aspects of the film were very impressive.\n\nNegative aspect:\n- Storyline: The review states that it was \"weak\", suggesting that the plot or narrative of the film was not well-developed or engaging.\n\nThis brief review contrasts the strong visual elements with the poor storytelling, implying that while the film may have been visually appealing, it lacked in narrative quality."
    }

