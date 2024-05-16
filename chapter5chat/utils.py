import logging
import pathlib
from langchain.schema import Document 
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.document_loaders import *
from langchain.document_loaders import UnstructuredEmailLoader
from typing import Any

class EPubReader(UnstructuredEPubLoader):
    def __int__(self, file_path:str | list[str], **kwargs: Any):
        super().__int__(file_path, **kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass
    
class DocumentLoader(object):
    supported_extensions ={
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EPubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

def load_document(file_path: str)-> list[Document]:
    ext = pathlib.Path(file_path).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(f"Invalid extension {ext}")
    loader = loader(file_path)
    docs = loader.load()
    return docs


