import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = "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"

def read_csv_into_dataframe(csv_name):
    df = pd.read_csv(csv_name)
    return df

def initialize_chatbot():
    data_frame = read_csv_into_dataframe(r"Renewable_Energy_Final123 1.csv")  # Replace "path_to_your_csv_file.csv" with the path to your CSV file
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    p_agent = create_pandas_dataframe_agent(llm=llm, df=data_frame, verbose=True, handle_parsing_errors=True)
    return p_agent

def get_chatbot_agent():
    if 'chatbot_agent' not in st.session_state:
        st.session_state.chatbot_agent = initialize_chatbot()
    return st.session_state.chatbot_agent

def main():
    st.title("Streamlit Chatbot")
    p_agent = get_chatbot_agent()

    user_input = st.text_input("You: ")
    if user_input:
        response = p_agent.run(user_input)
        st.text_area("Bot:", value=response, height=200)

if __name__ == "__main__":
    main()
