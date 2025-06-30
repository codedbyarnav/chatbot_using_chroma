# realme.py  – Streamlit app for RealMe.AI  (Chroma version)

# --------------------------------------------------------------------
# 0️⃣  Patch sqlite3 with pysqlite3 before anything else touches it
# --------------------------------------------------------------------
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # If pysqlite3 isn’t available the app will fall back to the
    # system sqlite3, but Chroma will likely fail – warn the dev.
    import warnings, sqlite3, sys
    warnings.warn(f"Using system sqlite3 ({sqlite3.sqlite_version}); "
                  "Chroma may require >=3.35")

# --------------------------------------------------------------------
# 1️⃣  Standard imports
# --------------------------------------------------------------------
import os
from pathlib import Path
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationEntityMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --------------------------------------------------------------------
# 2️⃣  Environment / constants
# --------------------------------------------------------------------
st.set_page_config(page_title="RealMe.AI - Ask Arnav", page_icon="🧠")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment variables.")
    st.stop()

DATA_PATH          = "data/"
VECTOR_STORE_PATH  = "vectorstore/db_chroma"

PROMPT_TEMPLATE = """
You are Arnav Atri's AI twin. ...
[template unchanged]
"""

# --------------------------------------------------------------------
# 3️⃣  Embeddings
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# --------------------------------------------------------------------
# 4️⃣  Build vector‑store in‑cloud if missing
# --------------------------------------------------------------------
def build_chroma_if_needed():
    db_folder = Path(VECTOR_STORE_PATH)
    if db_folder.exists() and any(db_folder.iterdir()):
        return  # already there ✔️

    st.warning("Vector store not found – building it now (first‑run only)…")

    # Lazy imports so we don’t pay startup cost on each run
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader     = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents  = loader.load()
    splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks     = splitter.split_documents(documents)
    embeddings = load_embeddings()

    db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
    db.persist()

    st.success("Vector store built and persisted! (reload page if it still shows spinner)")

# --------------------------------------------------------------------
# 5️⃣  Vector‑store loader
# --------------------------------------------------------------------
def load_vectorstore(embeddings):
    return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

# --------------------------------------------------------------------
# 6️⃣  Stream‑handler for LLM streaming
# --------------------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        if token and token.strip().lower() != "none":
            self.text += token
            self.container.markdown(self.text + "▌")

# --------------------------------------------------------------------
# 7️⃣  Build RAG + memory chain
# --------------------------------------------------------------------
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
        ai_prefix="I",
    )

    embeddings = load_embeddings()
    vector_db  = load_vectorstore(embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "history", "input"],
        template=PROMPT_TEMPLATE,
    )

    class RetrieverWrapper:
        def __init__(self, retriever):
            self.retriever = retriever

        def get_context(self, query: str) -> str:
            docs = self.retriever.get_relevant_documents(query)
            return "\n\n".join(d.page_content for d in docs)

    retriever_wrapper = RetrieverWrapper(vector_db.as_retriever())

    class RAGEntityChain(LLMChain):
        def invoke(self, inputs):
            inputs["context"] = retriever_wrapper.get_context(inputs["input"])
            return super().invoke(inputs)

    return RAGEntityChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# --------------------------------------------------------------------
# 8️⃣  UI
# --------------------------------------------------------------------
st.markdown("<h1 style='text-align:center'>🧠 RealMe.AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray'>Ask anything about Arnav Atri</h4>", unsafe_allow_html=True)
st.divider()

# Build vectorstore if needed on first cloud run
build_chroma_if_needed()

# Initialise chain
if "chat_chain" not in st.session_state:
    dummy_container = st.empty()
    st.session_state.chat_chain = get_rag_entity_chain(StreamHandler(dummy_container))

# Render chat history
for msg in st.session_state.chat_chain.memory.chat_memory.messages:
    role   = "user" if msg.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# New message
user_input = st.chat_input("Ask Arnav anything…")
if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    placeholder = st.empty()
    handler     = StreamHandler(placeholder)

    chain                       = st.session_state.chat_chain
    chain.llm.callbacks         = [handler]        # swap live handler
    chain.invoke({"input": user_input})            # stream

    # Strip “None” artifacts
    placeholder.markdown(handler.text.strip().replace("None", ""))

# --------------------------------------------------------------------
# 9️⃣  Footer
# --------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top:30px" />
    <div style="text-align:center;font-size:16px">
        🤝 <strong>Let’s connect</strong><br/>
        <a href="https://www.linkedin.com/in/arnav-atri-315547347/" target="_blank" style="margin:0 20px">🔗 LinkedIn</a> |
        <a href="mailto:arnavatri5@gmail.com?subject=Hello%20Arnav&body=I%20found%20your%20RealMe.AI%20chatbot%20amazing!" target="_blank">📧 Email</a>
    </div>
    """,
    unsafe_allow_html=True,
)
