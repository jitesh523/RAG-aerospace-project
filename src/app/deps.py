from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from src.config import Config

def build_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    vs = FAISS.load_local("./faiss_store", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 25})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
