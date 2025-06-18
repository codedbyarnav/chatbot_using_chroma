# streamlit_app.py
import os
os.environ["CHROMA_TELEMETRY"] = "False"

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # ✅ Changed from FAISS

# Load API key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

# Constants
VECTOR_STORE_PATH = "vectorstore/db_chroma"  # ✅ Changed path to Chroma DB
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

# Embeddings and vector store loader
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(embeddings):
    return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

# Streaming handler to stream inside chat bubble
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        if token and token.strip().lower() != "none":
            self.text += token
            self.container.markdown(self.text + "▌")

# Chain builder with Entity Memory and RAG
def get_rag_entity_chain(stream_handler):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler],
    )

    memory = ConversationEntityMemory(
        llm=llm,
        memory_key="history",
        input_key="input",
        return_messages=True,
        human_prefix="you",
        ai_prefix="I"
    )

    embeddings = load_embeddings()
    vector_db = load_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "history", "input"],
        template=PROMPT_TEMPLATE,
    )

    class RetrieverWrapper:
        def __init__(self, retriever):
            self.retriever = retriever

        def get_context(self, query):
            docs = self.retriever.get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in docs])

    retriever_wrapper = RetrieverWrapper(vector_db.as_retriever())

    class RAGEntityChain(LLMChain):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def invoke(self, inputs):
            context = retriever_wrapper.get_context(inputs["input"])
            inputs["context"] = context
            return super().invoke(inputs)

    return RAGEntityChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# Header
st.markdown("<h1 style='text-align: center;'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Initialize chat chain once
if "chat_chain" not in st.session_state:
    dummy_container = st.empty()
    stream_handler = StreamHandler(dummy_container)
    st.session_state.chat_chain = get_rag_entity_chain(stream_handler)

# Show previous messages
for message in st.session_state.chat_chain.memory.chat_memory.messages:
    with st.chat_message("user" if message.type == "human" else "assistant",
                         avatar="🧑‍💻" if message.type == "human" else "🤖"):
        st.markdown(message.content)

# Input box
user_input = st.chat_input("Ask Arnav anything...")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖") as assistant_container:
        stream_placeholder = st.empty()
        stream_handler = StreamHandler(stream_placeholder)

        # Use the existing memory-enabled chain
        chat_chain = st.session_state.chat_chain
        chat_chain.llm.callbacks = [stream_handler]

        # Streaming response (invoke returns None)
        chat_chain.invoke({"input": user_input})

        # Re-render the clean final output (removes trailing ▌ and avoids "None")
        final_response = stream_handler.text.strip().replace("None", "").replace("NONE", "")
        stream_placeholder.markdown(final_response)

# Footer
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; font-size: 16px;">
🤝 <strong>Let’s connect</strong><br>
<a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="text-decoration: none; margin: 0 20px;">
🔗 LinkedIn
</a>
|
<a href="https://mail.google.com/mail/?view=cm&fs=1&to=arnavatri5@gmail.com&su=Hello+Arnav&body=I+found+your+RealMe.AI+chatbot+amazing!" target="_blank" style="text-decoration: none;">
📧 Email
</a>
</div>
""", unsafe_allow_html=True)
