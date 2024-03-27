import langchain
import json
import requests
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import PubMedRetriever
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "PubMed"

# RESULTS_PER_QUESTION = 3

# ddg_search = DuckDuckGoSearchAPIWrapper()

# def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
#     results = ddg_search.results(query, num_results)
#     return [r["link"] for r in results]


# SUMMARY_TEMPLATE = """{text} 
# -----------
# Using the above text, answer in short the following question: 
# > {question}
# -----------
# if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
# SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


# def scrape_text(url: str):
#     # Send a GET request to the webpage
#     try:
#         response = requests.get(url)

#         # Check if the request was successful
#         if response.status_code == 200:
#             # Parse the content of the request with BeautifulSoup
#             soup = BeautifulSoup(response.text, "html.parser")

#             # Extract all text from the webpage
#             page_text = soup.get_text(separator=" ", strip=True)

#             # Print the extracted text
#             return page_text
#         else:
#             return f"Failed to retrieve the webpage: Status code {response.status_code}"
#     except Exception as e:
#         print(e)
#         return f"Failed to retrieve the webpage: {e}"


# url = "https://www.worldhistory.org/herodotus/"

# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary = RunnablePassthrough.assign(
#     text=lambda x: scrape_text(x["url"])[:10000]
# ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
# ) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

# web_search_chain = RunnablePassthrough.assign(
#     urls = lambda x: web_search(x["question"])
# ) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


##This is for PubMed
from langchain_community.retrievers import PubMedRetriever
retriever = PubMedRetriever()

SUMMARY_TEMPLATE = """{doc} 
 
 -----------
 
 Using the above text, answer in short the following question: 
 
 > {question}
 
 -----------
 if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available.""" 

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
 
 
scrape_and_summarize_chain = RunnablePassthrough.assign(
     summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
 ) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")
 
web_search_chain = RunnablePassthrough.assign(
     docs = lambda x: retriever.get_relevant_documents(x["question"])
 )| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()

## This is for Arxiv

# from langchain.retrievers import ArxivRetriever
# 
# retriever = ArxivRetriever()
# SUMMARY_TEMPLATE = """{doc} 
# 
# -----------
# 
# Using the above text, answer in short the following question: 
# 
# > {question}
# 
# -----------
# if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
# SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
# 
# 
# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
# ) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")
# 
# web_search_chain = RunnablePassthrough.assign(
#     docs = lambda x: retriever.get_summaries_as_docs(x["question"])
# )| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()



SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 research queries to search in PubMed that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

# ```prompt
# To generate three research queries for PubMed that form an objective opinion from the provided question, please replace {question} with your specific topic of interest. For example, if your topic is "the impact of diet on cardiovascular health," your input should look like this: "the impact of diet on cardiovascular health". Here is the prompt template you can use:

# ["{question} systematic review", "{question} meta-analysis", "{question} randomized controlled trial"]

# Remember to replace {question} with your actual research question without the quotation marks. For instance:

# ["impact of diet on cardiovascular health systematic review", "impact of diet on cardiovascular health meta-analysis", "impact of diet on cardiovascular health randomized controlled trial"]

# Please provide your research question to proceed with generating the PubMed queries.
# ```

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/PubMed-assistant",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    
#http://localhost:8000/research-assistant/playground/ in case of general recearch
#http://localhost:8000/PubMed-assistant/playground/ in case of PubMed