from typing import Annotated 
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START,END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


llm=ChatOpenAI(temperature=0)

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state:State):
        return {"messages":[llm.invoke(state["messages"])]}
    
    ## node 

    graph_workflow.add_node('agent',call_model)

    ## adding edges 
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)

    agent=graph_workflow.compile()
    return agent

agent=make_default_graph()
