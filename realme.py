import os, sys, warnings
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    import sqlite3
    warnings.warn(f"using system sqlite3 {sqlite3.sqlite_version}; Chroma may need ≥3.35")

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found"); st.stop()

VECTOR_STORE_PATH = "vectorstore/db_chroma"

PROMPT_TEMPLATE = """
You are Arnav Atri's AI twin. You will carry a memory of Arnav's life and conversations with users.

Maintain a friendly tone, respond with Arnav's perspective, and use remembered facts about people, places, or preferences as the chat continues.
Use only the information from the documents below if relevant. If unsure, say so honestly.
Never respond with or include the word "None" in any form.

Context:
{context}

Current conversation history:
{history}

New user input:
{input}

Reply as Arnav:
"""

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def load_vectorstore(embeddings):
    return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container, self.text = container, ""
    def on_llm_new_token(self, token, **_):
        if token and token.strip().lower() != "none":
            self.text += token
            self.container.markdown(self.text + "▌")

def get_rag_entity_chain(handler):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY,
                     streaming=True, callbacks=[handler])

    memory = ConversationEntityMemory(llm=llm, memory_key="history",
                                      input_key="input", return_messages=True,
                                      human_prefix="you", ai_prefix="I")

    embeddings = load_embeddings()
    vector_db  = load_vectorstore(embeddings)
    prompt     = PromptTemplate(input_variables=["context", "history", "input"],
                                template=PROMPT_TEMPLATE)

    class RetrieverWrapper:
        def __init__(self, retriever): self.retriever = retriever
        def get_context(self, q): return "\n\n".join(d.page_content for d in self.retriever.get_relevant_documents(q))

    retriever_wrapper = RetrieverWrapper(vector_db.as_retriever())

    class RAGChain(LLMChain):
        def invoke(self, inputs):
            inputs["context"] = retriever_wrapper.get_context(inputs["input"])
            final_prompt = prompt.format(**inputs)
            print("🔍 User Input:", inputs["input"])
            print("📄 Retrieved Context:\n", inputs["context"])
            print("🧾 Final Prompt Sent to LLM:\n", final_prompt)
            return super().invoke(inputs)

    return RAGChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

st.markdown("<h1 style='text-align:center'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = get_rag_entity_chain(StreamHandler(st.empty()))

for m in st.session_state.chat_chain.memory.chat_memory.messages:
    role = "user" if m.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(m.content)

user_input = st.chat_input("Ask Arnav anything…")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"): st.markdown(user_input)
    placeholder = st.empty(); handler = StreamHandler(placeholder)
    chain = st.session_state.chat_chain; chain.llm.callbacks = [handler]
    chain.invoke({"input": user_input})
    placeholder.markdown(handler.text.strip().replace("None", ""))

st.markdown("""
<hr style="margin-top:30px"/>
<div style="text-align:center;font-size:16px">
🤝 <strong>Let’s connect</strong><br/>
<a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="margin:0 20px">🔗 LinkedIn</a> |
<a href="mailto:arnavatri5@gmail.com?subject=Hello%20Arnav&body=I%20found%20your%20RealMe.AI%20chatbot%20amazing!" target="_blank">📧 Email</a>
</div>
""", unsafe_allow_html=True)
