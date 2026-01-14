# MacroBot

## Setup Instructions
### 1. Install Python
Make sure you have **Python 3.10** installed on your system.

### 2. Create Virtual Environment
Run the following command in your terminal to create the environment and install all dependencies.
```
mmake create_env
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
### To run the agent:
```
make run_agent
```

### To run the streamlit application:
```
make run_agent_ui
```

