from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search,scrap_url
import os
from rich import print
from dotenv import load_dotenv

load_dotenv()

#model Setup
llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0, api_key=os.getenv("GROQ_API_KEY"))

#1st Agent
def search_agent():
    return create_agent(
        model = llm,
        tools=[web_search]
    )


#2nd Agent
def reader_agent():
    return create_agent(
        model=llm,
        tools=[scrap_url]
    )

# Writer Chain

write_prompt=ChatPromptTemplate.from_messages([
    ("system","Ypu are an expert research writer. write clear, structured and insightful reports"),
    ("human","""Write a detailed research report on the topic below
    Topic: {topic}
    research Gathered:
    {research}

    Structured the report as:
    - Introduction
    - Key finding (Minimum 3 Well explained points)
    - Conclusion
    - Source (List All The  URLs found in the research)
    
    Be detailed, factual and professional.
    """),
])


writer_chain = write_prompt | llm | StrOutputParser()

# Critic_chain 
critic_prompts = ChatPromptTemplate.from_messages([
    ('system',"you are a sharp and contructive research critic. Be Honest and specific."),
    ('human',"""Review the research report below and evaluate it strictly.
    Report:{report}

    Response in the exct format:

    Score : x/10

    Strengths:
    -...
    -...

    Areas To Improve:
    -...
    -...
    
    One Line Verdict:
    ...
    
    """)
])


critic_chain = critic_prompts | llm | StrOutputParser()