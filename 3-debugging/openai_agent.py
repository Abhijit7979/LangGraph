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

def make_tool_graph():
    
    @tool
    def add(a:float ,b: float):
        """Adds two numbers"""
        return a+b
    tool_node=ToolNode([add])

    model_with_tools=llm.bind_tools([add])
    def call_model(state:State):
        return {"messages":[model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state:State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END
    builder=StateGraph(State)
    builder.add_node("agent",call_model)
    builder.add_node("tools",tool_node)
    builder.add_edge("tools","agent")
    builder.add_edge(START,"agent")
    builder.add_conditional_edges("agent",should_continue,"tools")
    

    agent=builder.compile()

    return agent
def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = llm.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

agent=make_alternative_graph()
