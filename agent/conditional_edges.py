"""
This file contains the conditional edges for the agent.
"""

class MacroeconomicsConditionalEdges:
    @staticmethod
    def classify_user_message_decision(state) -> str:
        """
        This function decides which node to go to based on the user message.
        We have two categories: Macroeconomics and General.
        """
        if state.category == "MACROECONOMICS":
            return "enhanced_macroeconomics_response"
        elif state.category == "GENERAL":
            return "generate_general_response"
        else:
            return "fallback"
