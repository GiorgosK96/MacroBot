"""
This file contains the workflow for the Macroeconomics Agent.
"""

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from agent.conditional_edges import MacroeconomicsConditionalEdges
from agent.graph_state import GraphState
from agent.nodes import MacroeconomicsNodes
from logger import logger


class MacroeconomicsWorkflow:
    """
    This class contains the workflow for the Macroeconomics Agent.
    """

    def __init__(self):
        """
        Initialize the Macroeconomics Workflow.
        """
        self.history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(chat_memory=self.history, return_messages=True)
        self.nodes = MacroeconomicsNodes()
        self.conditional_edges = MacroeconomicsConditionalEdges()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """
        Build the workflow for the Macroeconomics Agent.
        """
        try:
            # Initialize the workflow
            workflow = StateGraph(GraphState)

            # Add nodes to the workflow
            workflow.add_node("classify_user_message", self.nodes.classify_user_message)
            workflow.add_node("generate_general_response", self.nodes.generate_general_response)
            workflow.add_node("enhanced_macroeconomics_response", self.nodes.enhanced_macroeconomics_response)
            # Add starting edge to the workflow
            workflow.add_edge(START, "classify_user_message")
            # Add conditional edge to the workflow depending on the user message
            workflow.add_conditional_edges("classify_user_message", self.conditional_edges.classify_user_message_decision)
            # Add ending edges to the workflow
            workflow.add_edge("generate_general_response", END)
            workflow.add_edge("enhanced_macroeconomics_response", END) 

            # Compile the workflow
            memory = MemorySaver()
            graph = workflow.compile(checkpointer=memory)

        except Exception as e:
            logger.error(f"Error while building the workflow: {e}")
            raise e

        return graph
