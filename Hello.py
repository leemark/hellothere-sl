__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import re
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import DeepInfra
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader 
from langchain.indexes import VectorstoreIndexCreator

st.title('🐯 Ask a Tiger.')

openai_api_key = st.secrets["OPENAI_API_KEY"]
DEEPINFRA_API_TOKEN = st.secrets["DEEPINFRA_API_TOKEN"]

def generate_response(input_text):
  llm = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    max_tokens=2048,
    temperature=0.9, 
    openai_api_key=openai_api_key
  )
  llm_di = DeepInfra(model_id="cognitivecomputations/dolphin-2.6-mixtral-8x7b")
  llm_di.model_kwargs = {
      "temperature": 0.7,
      "repetition_penalty": 1,
      "max_new_tokens": 2048,
      "top_p": 0.9
  }
  llm_query = llm_di(f"Given the following question, what search term would you use to search for the answer? Respond only with the search terms, nothing else. {input_text}")
  st.info(llm_query)
  srch_query = f"{llm_query} site:coloradocollege.edu"
  wrapper = DuckDuckGoSearchAPIWrapper(max_results=12)
  search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
  context = search.run(srch_query)
  links = re.findall(r'https?://\S+', context)
  cleaned_links = [link.rstrip("],") for link in links]

  loader_list = []
  for i in cleaned_links:
    loader_list.append(WebBaseLoader(i))
  index = VectorstoreIndexCreator().from_loaders(loader_list)
  
  prompt = f'''
    You are a helpful and friendly CC student who is answering questions about Colorado College (aka CC). 
    Given all of the context, please provide a comprehensive answer to the user's question: {input_text} 
    Make sure that the answer would be helpful to a prospective student. 
  '''

  ans = index.query(question=prompt, llm=llm)
  st.info(ans)

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What is the Block Plan?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)