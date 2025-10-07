# Faculty-Workload-Timetable-AI-Agent

Minimal runnable MVP with a mocked LLM for demo.

Structure:
- data/: sample CSVs and policies
- src/: core code (preprocess, tools, agent, UI)
- tests/: basic tests
- requirements.txt, Dockerfile

To run (locally):
1. python3 -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. streamlit run src/ui_streamlit.py
