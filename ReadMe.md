# MacroBot

## What MacroBot Does

MacroBot is an intelligent AI assistant specialized in macroeconomics, designed to help students and professionals understand complex economic concepts. Using advanced Retrieval-Augmented Generation (RAG) technology, MacroBot provides accurate, contextual answers by drawing from a comprehensive collection of authoritative macroeconomics textbooks, including:

- Blanchard & Johnson - Macroeconomics (6th Edition)
- Mankiw - Macroeconomics (7th Edition)
- Mankiw - Principles of Economics (7th Edition)
- Krugman & Wells - Macroeconomics
- Dornbusch, Fischer & Startz - Macroeconomics (11th Edition)
- Mishkin & Serletis - The Economics of Money, Banking & Financial Markets
- OpenStax - Principles of Macroeconomics

### Key Features:
- **Intelligent Classification**: Automatically distinguishes between macroeconomics-related questions and general queries
- **RAG-Powered Responses**: Retrieves relevant information from multiple textbooks using FAISS vector search
- **Conversation History**: Maintains context across multiple questions for more coherent interactions
- **Multi-Source Citations**: Provides answers grounded in academic sources with proper attribution
- **Streamlit UI**: User-friendly web interface for seamless interaction
- **Bilingual Support**: Supports both English and Greek language queries

## Setup Instructions
### 1. Install Python
Make sure you have **Python 3.10** installed on your system.

### 2. Create Virtual Environment
Run the following command in your terminal to create the environment and install all dependencies.
```
make create_env
```

### 3. Activate your Virtual Environment

```
source .venv/bin/activate
```

### 4. Update your agent.env file with your open api key.
```
[AGENT]
OPENAI_API_KEY = <your-api-key>
```

## Agent

### To run the streamlit agent application:
```
make run_agent_ui
```

