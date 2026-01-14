# Create and set up the virtual environment with Python 3.10
create_env:
	python3.10 -m venv .venv
	pip install --upgrade pip
	pip install -r requirements.txt


# Run the agent
run_agent:
	python -m agent.main

# Run the streamlit application
run_agent_ui:
	streamlit run agent/app.py