import pathlib
import sys
import streamlit as st
import json
import csv
import os
from datetime import datetime
from langchain_core.messages import HumanMessage


# Setup paths
ROOT_FOLDER = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(ROOT_FOLDER)

from agent.config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE
from agent.graph import MacroeconomicsWorkflow
from agent.graph_state import GraphState


# UI setup
st.set_page_config(page_title="MacroBot", layout="wide")

# Create main layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("MacroBot", divider="blue")

with col2:
    st.write("")  # Spacing


@st.cache_resource
def setup():
    app = MacroeconomicsWorkflow()
    state = GraphState(messages=[])
    return app, state

app, state = setup()


# CSV Export Functions
def create_conversations_folder():
    """Create conversations folder if it doesn't exist"""
    folder_path = "conversations"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_conversation_to_csv():
    """Save current conversation to CSV file"""
    if len(st.session_state.messages) <= 1:  # Only initial message
        st.sidebar.warning("Δεν υπάρχει συνομιλία για αποθήκευση")
        return None
    
    try:
        folder_path = create_conversations_folder()
        conversation_number = len([f for f in os.listdir(folder_path) if f.endswith('.csv')]) + 1
        filename = f"{folder_path}/conversation_{conversation_number:03d}.csv"
        
        # Prepare conversation data
        conversation_data = []
        question_id = 0
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                question_id += 1
                user_message = message["content"]
                
                # Find corresponding assistant response
                assistant_response = ""
                
                
                if i + 1 < len(st.session_state.messages):
                    next_message = st.session_state.messages[i + 1]
                    if next_message["role"] == "assistant":
                        assistant_response = next_message["content"]
                        
                
                conversation_data.append({
                    "Question_ID": question_id,
                    "User_Question": user_message,
                    "Assistant_Response": assistant_response,
                })
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Question_ID", "User_Question", "Assistant_Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in conversation_data:
                writer.writerow(row)
        
        st.sidebar.success(f"Η συνομιλία αποθηκεύτηκε επιτυχώς!")
        return filename
        
    except Exception as e:
        st.sidebar.error(f"Σφάλμα αποθήκευσης: {e}")
        return None

def reset_conversation():
    """Reset conversation to initial state"""
    st.session_state.messages = [
        {"role": "assistant", "content": "Γεια, είμαι το MacroBot, ο βοηθός σου για θέματα Μακροοικονομίας. Πώς μπορώ να σε βοηθήσω;"}
    ]
    # Reset the graph state as well
    state.messages = []

# Sidebar Controls
st.sidebar.title("Έλεγχος Συνομιλίας")

# New Chat Button
if st.sidebar.button("Νέα Συνομιλία", use_container_width=True, type="primary"):
    # Απλά reset - χωρίς auto-save
    reset_conversation()
    st.rerun()

# Manual save button
st.sidebar.markdown("---")
if st.sidebar.button("Αποθήκευση Συνομιλίας", use_container_width=True):
    save_conversation_to_csv()

# Info section
st.sidebar.markdown("---")
st.sidebar.markdown("Οδηγίες Χρήσης")
st.sidebar.markdown("""
**Πώς να χρησιμοποιήσεις το MacroBot:**
1. Γράψε την ερώτησή σου στο πεδίο εισαγωγής
2. Αν είναι σχετική με τη μακροοικονομία, θα λάβεις τεκμηριωμένη απάντηση με πηγές
3. Αν είναι γενική ερώτηση, θα λάβεις μια φιλική απάντηση και κατεύθυνση

**Συμβουλές:**
- Πάτησε "Αποθήκευση Συνομιλίας" για να αποθηκεύσεις τη συνομιλία σου
- Χρησιμοποίησε "Νέα Συνομιλία" για να ξεκινήσεις από την αρχή
""")


# Initialize chat history
if "messages" not in st.session_state:
    reset_conversation()

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("type") == "graph":
            # Handle graph messages if needed
            st.markdown("Graph visualization would go here")
        else:
            st.markdown(message["content"])

# Handle response rendering
def handle_assistant_response(assistant_response):
    try:
        assistant_response = assistant_response.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(assistant_response)
        if "graph" in parsed:
            st.session_state.messages.append({"role": "assistant", "content": parsed, "type": "graph"})
            st.rerun()
        else:
            response_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            state.messages.append(HumanMessage(content=assistant_response))
    except Exception as e:
        response_placeholder.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        state.messages.append(HumanMessage(content=assistant_response))

# User input handling
if user_input := st.chat_input("Εισάγετε την ερώτησή σας εδώ..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    config_local = {"configurable": {"thread_id": "1"}}
    state.messages.append(HumanMessage(content=user_input))

    # Process the assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("`Το MacroBot πληκτρολογεί...`")

        try:
            for event in app.workflow.stream(state, config=config_local):
                for value in event.values():
                    assistant_response = value["messages"][-1].content
                    assistant_response = assistant_response.replace("```json", "")
                    assistant_response = assistant_response.replace("```", "")
                    if assistant_response != user_input:
                        handle_assistant_response(assistant_response)

        except Exception as e:
            response_placeholder.markdown(f"Error: {e}")

# Auto-save on every response (optional)
# Uncomment if you want to auto-save after each question-answer pair
# if len(st.session_state.messages) > 1 and st.session_state.messages[-1]["role"] == "assistant":
#     # Auto-save after each complete Q&A
#     pass