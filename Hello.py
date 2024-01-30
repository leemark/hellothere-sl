__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import re
from langchain_openai import OpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader 
from langchain.indexes import VectorstoreIndexCreator

st.title('🐯 Tiger Talk')

openai_api_key = st.secrets["OPENAI_API_KEY"]

def generate_response(input_text):
  # llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  llm = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    max_tokens=2048,
    temperature=0.7, 
    openai_api_key=openai_api_key
  )
  srch_query = f"{input_text} site:coloradocollege.edu"
  wrapper = DuckDuckGoSearchAPIWrapper(max_results=12)
  search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
  context = search.run(srch_query)
  links = re.findall(r'https?://\S+', context)
  cleaned_links = [link.rstrip("],") for link in links]
  loader_list = []
  for i in cleaned_links:
    loader_list.append(WebBaseLoader(i))
  index = VectorstoreIndexCreator().from_loaders(loader_list)
  prompt = '''
    Given the context, please provide an answer to {input_text}
  '''
  #st.info(llm(input_text))
  st.info(prompt)
  ans = index.query(question=prompt, llm=llm)
  st.info(ans)

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What is the Block Plan?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)