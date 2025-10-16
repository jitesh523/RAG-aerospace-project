from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from src.config import Config

def build_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    backend = Config.RETRIEVER_BACKEND
    if backend == "milvus":
        # Build Milvus-backed vector store retriever
        vs = LC_Milvus(
            embedding_function=embeddings,
            collection_name=Config.MILVUS_COLLECTION,
            connection_args={"host": Config.MILVUS_HOST, "port": str(Config.MILVUS_PORT)},
            auto_id=False,
        )
    else:
        # Default to FAISS
        vs = FAISS.load_local("./faiss_store", embeddings=embeddings)

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.RETRIEVER_K, "fetch_k": Config.RETRIEVER_FETCH_K},
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
