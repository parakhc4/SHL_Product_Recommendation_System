import streamlit as st
import os
from dotenv import load_dotenv
MONSTER_API_KEY = os.environ.get("MONSTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")