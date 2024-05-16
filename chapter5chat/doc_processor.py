from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
import os
import tempfile
from utils import load_document
from dotenv import load_dotenv

def configure_retriever(
    docs: list[Document], 
    use_compression: bool = False)-> BaseRetriever:
    """ 
    Retrive document by maximum marginal relevance.
    1. Split document
    2. Create Embeddings
    3. Store the documents
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Use DocArrayInMemorySearch as in memory storage
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", 
                                      search_kwargs={
                                          "k": 2, 
                                          "fetch_k": 4,
                                          "include_metadata": True},)
    if not use_compression:
        return retriever
    #Retrieval can be improved by contextual compression 
    # a technique where retrieved document is compressed and irrelevant
    # document is filter out.  Compress document based on context of given query
    # Few options for contextual compression
    # 1. LLMChainCompression
    # 2. LLMChainFilter
    # 3. EmbeddingFilter: Applies similarity filter based on the doc and query the terms of embeddings

    embddings_filter = EmbeddingsFilter(
        embeddings= embeddings, similarity_threshold=.8
    )
    
    return ContextualCompressionRetriever(
        base_compressor=embddings_filter,
        base_retriever=retriever
    )
    

# Chain for having conversation based on retrieved documents
def configure_chain(retriever: BaseRetriever)-> Chain:
    load_dotenv()
    # set up memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # set up LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    # Use llm to geneate response based on retrieved document
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=4000)
    return chain


# Load file from a directory and Document Loader to load the document.
# Call retriever and Use LLM to have conversation based on retrieved document
def configure_conversation(uploaded_files):
    docs =[]
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))
    
    retriever = configure_retriever(docs=docs)    
    chain = configure_chain(retriever) 
    return chain

