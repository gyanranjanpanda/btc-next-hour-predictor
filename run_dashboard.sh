#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
streamlit run src/interfaces/dashboard.py
