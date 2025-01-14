from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import shutil
import os

DATA_PATH = "data"

##################################3333

from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_content) -> None:
        """Initialize the loader with a file path.

        Args:
            file: The path to the file to load.
        """
        self.file = file_content

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        for file_ in self.file:
            if file_["content"]:
                yield Document(
                    page_content=file_["content"],
                    metadata={"snippet": file_["snippet"], "source": file_["title"], "link": file_["link"]},
                )

#################################33

def load_documents(path=DATA_PATH):
    loader = DirectoryLoader(path)
    return loader.load()

def split_text(documents, chunk_size=1000, chunk_overlap=500, length_function=len, add_start_index=True):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function, add_start_index=add_start_index)
    chunks = text_splitter.split_documents(documents)
    # print(chunks[0].page_content, chunks[0].metadata)
    # print(f"Split Document of length {len(documents)} into {len(chunks)} chunks")
    return chunks

def clear_dir(path=DATA_PATH):
    if os.path.exists(path):
        shutil.rmtree(path)

def create_dir(path=DATA_PATH):
    if not os.path.exists(path):
        os.makedirs(path)