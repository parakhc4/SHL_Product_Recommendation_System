import time
from openai import OpenAI
import streamlit as st
MONSTER_API_KEY = st.secrets["MONSTER_API_KEY"]
from .config import MONSTER_MODEL_MAP, LLM_NAME

def setup_monster_api():
    return OpenAI(
        base_url="https://llm.monsterapi.ai/v1/",
        api_key=(MONSTER_API_KEY),
    )

def call_monster_api(user_input, context, client, retries=3, wait_time=5):
    attempt = 0
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model=MONSTER_MODEL_MAP[LLM_NAME],
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input},
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(wait_time)
                attempt += 1
            else:
                break
    return "Could not generate a response."
