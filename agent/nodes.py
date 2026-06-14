from agent.config import MODEL_NAME, TEMPERATURE, OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from agent.prompts import classify_user_message_prompt, generate_general_response_prompt, generate_macroeconomics_response_prompt
from logger import logger


class MacroeconomicsNodes:
    """
    This class contains the nodes for the Macroeconomics Agent.
    Updated με RAG integration και detailed logging
    """
    
    def __init__(self):
        """
        Initialize the Macroeconomics Nodes.
        """
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
        
        # Re-enable RAG με detailed logging
        try:
            logger.info("Initializing RAG system...")
            from agent.rag_system import MacroeconomicsRAG
            self.rag_system = MacroeconomicsRAG()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            import traceback
            logger.error(f"RAG init traceback: {traceback.format_exc()}")
            self.rag_system = None

    def classify_user_message(self, state):
        """Classifies the user message into categories Macroeconomics or General."""

        try:
            state.category = None
            conversation_history = [("User: " if isinstance(msg, HumanMessage) else "Bot: ") + 
                                    msg.content for msg in state.messages][:-1]
            last_user_message = state.messages[-1].content

            logger.info(f"Classifying user's message: {last_user_message}")

            prompt = classify_user_message_prompt.format(
                conversation_history=conversation_history,
                last_user_message=last_user_message
                )
        
            classification = self.llm.invoke(prompt)
            state.category = classification.content
            logger.info(f"Classification: {state.category}")
            state.question = last_user_message
        except Exception as e:
            logger.error(f"Error in classify_user_message: {e}")
            
        return state
    

    def generate_general_response(self, state):
        """Generates a response to the user's general message."""
        try:
            conversation_history = [("User: " if isinstance(msg, HumanMessage) else "Bot: ") + 
                                    msg.content for msg in state.messages][:-1]
            last_user_message = state.messages[-1].content

            logger.info(f"Generate a response to the user's general message: {last_user_message}")

            prompt = generate_general_response_prompt.format(
                conversation_history=conversation_history,
                last_user_message=last_user_message
                )
        
            response = self.llm.invoke(prompt)
            logger.info(f"Response: {response}")
            state.question = last_user_message
            state.messages.append(response)
        except Exception as e:
            logger.error(f"Error in responding the general message: {e}")
            
        return state

    def enhanced_macroeconomics_response(self, state):
        """
        Enhanced macroeconomics response με RAG + GPT fallback
        ΜΕ ΠΛΗΡΕΣ ΙΣΤΟΡΙΚΟ ΣΥΝΟΜΙΛΙΑΣ για μέγιστο context
        """
        import time
        
        try:
            conversation_history = [("User: " if isinstance(msg, HumanMessage) else "Bot: ") + 
                                    msg.content for msg in state.messages][:-1]
            last_user_message = state.messages[-1].content

            logger.info(f"Enhanced macro response for: {last_user_message}")
            logger.info(f"RAG system available: {self.rag_system is not None}")
            
            # Προσπάθεια με RAG πρώτα
            if self.rag_system:
                logger.info("Starting RAG query...")
                start_time = time.time()
                
                try:
                    # Για το FAISS query: μόνο user μηνύματα (όχι bot απαντήσεις)
                    user_history = [f"User: {msg.content}" for msg in state.messages[:-1]
                                    if isinstance(msg, HumanMessage)]

                    if user_history:
                        query_with_context = f"""{chr(10).join(user_history)}
Τρέχουσα ερώτηση: {last_user_message}"""
                        logger.info(f"Using query with context ({len(user_history)} messages)")
                    else:
                        query_with_context = last_user_message
                        logger.info("No history - using direct query")

                    rag_result = self.rag_system.query(query_with_context)
                    
                    query_time = time.time() - start_time
                    logger.info(f"RAG query completed in {query_time:.2f} seconds")
                    logger.info(f"RAG result success: {rag_result.get('success', False)}")
                    logger.info(f"RAG sources found: {rag_result.get('num_sources', 0)}")
                    
                    # Έλεγχος ποιότητας RAG αποτελέσματος
                    logger.info("Checking RAG quality...")
                    is_good = self.rag_system.is_rag_result_good(rag_result)
                    logger.info(f"RAG quality check: {is_good}")
                    
                    if is_good:
                        # Χρήση RAG απάντησης
                        logger.info("Using RAG response")
                        response_content = rag_result["result"]
                        sources_info = self.rag_system.get_sources_info(rag_result.get("source_documents", []))
                        
                        final_response = f"{response_content}\n\n---\n**Απάντηση από ακαδημαϊκά συγγράμματα**\n{sources_info}"
                    else:
                        # Fallback σε κανονικό GPT
                        logger.info("RAG result not good enough, falling back to standard GPT")
                        logger.info("Starting standard GPT response...")
                        gpt_start = time.time()
                        
                        final_response = self._generate_standard_macro_response(conversation_history, last_user_message)
                        final_response += "\n\n---\n**Απάντηση από γενική γνώση AI** (δεν βρέθηκαν επαρκείς πληροφορίες στα συγγράμματα)"
                        
                        gpt_time = time.time() - gpt_start
                        logger.info(f"Standard GPT completed in {gpt_time:.2f} seconds")
                        
                except Exception as rag_error:
                    logger.error(f"RAG query failed: {rag_error}")
                    logger.info("Falling back to standard GPT due to RAG error")
                    
                    final_response = self._generate_standard_macro_response(conversation_history, last_user_message)
                    final_response += f"\n\n---\n**Σημείωση:** Υπήρξε πρόβλημα με το RAG σύστημα, χρησιμοποιήθηκε γενική γνώση."
            else:
                # Αν το RAG δεν είναι διαθέσιμο, χρήση κανονικού GPT
                logger.warning("RAG system not available, using standard response")
                logger.info("Starting standard GPT response...")
                gpt_start = time.time()
                
                final_response = self._generate_standard_macro_response(conversation_history, last_user_message)
                final_response += "\n\n---\n**Απάντηση από γενική γνώση AI** (το RAG σύστημα δεν είναι διαθέσιμο)"
                
                gpt_time = time.time() - gpt_start
                logger.info(f"Standard GPT completed in {gpt_time:.2f} seconds")

            logger.info("Creating response message...")
            response_message = AIMessage(content=final_response)
            state.messages.append(response_message)
            state.question = last_user_message
            
            logger.info("Enhanced macro response completed")
            return state

        except Exception as e:
            logger.error(f"Error in enhanced_macroeconomics_response: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback response
            try:
                fallback_response = self._generate_standard_macro_response(
                    [("User: " if isinstance(msg, HumanMessage) else "Bot: ") + 
                     msg.content for msg in state.messages][:-1],
                    state.messages[-1].content
                )
                fallback_response += f"\n\n---\n **Σημείωση:** Υπήρξε πρόβλημα με το σύστημα ({str(e)}), χρησιμοποιήθηκε γενική γνώση."
                
                response_message = AIMessage(content=fallback_response)
                state.messages.append(response_message)
                state.question = state.messages[-2].content
            except Exception as fallback_error:
                logger.error(f"Even fallback failed: {fallback_error}")
                error_message = AIMessage(content="Συγγνώμη, υπήρξε ένα πρόβλημα με την επεξεργασία της ερώτησής σου. Παρακαλώ δοκίμασε ξανά.")
                state.messages.append(error_message)
            
        return state

    def _generate_standard_macro_response(self, conversation_history, last_user_message):
        """
        Δημιουργεί κανονική μακροοικονομική απάντηση με GPT
        """
        try:
            logger.info("Generating standard macro response with GPT...")
            prompt = generate_macroeconomics_response_prompt.format(
                conversation_history=conversation_history,
                last_user_message=last_user_message
            )
            
            response = self.llm.invoke(prompt)
            logger.info("Standard macro response generated successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Error in standard macro response: {e}")
            return "Συγγνώμη, δεν μπόρεσα να επεξεργαστώ την ερώτησή σου αυτή τη στιγμή."

    def fallback(self, state):
        """Fallback response."""
        logger.info("Fallback response")
        state.messages.append(AIMessage(content="Δυστυχώς δεν ήταν δυνατό να βρω την απάντηση στην ερώτηση σου, παρακαλώ προσπάθησε να δώσεις μια πιο συγκεκριμένη ερώτηση."))
        return state