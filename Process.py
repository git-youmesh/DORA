from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import csv
from langchain_community.embeddings import HuggingFaceEmbeddings
loader = PyPDFLoader('dora.pdf')
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=200,  # Adjust as needed for context
        separators=["\n\n", "\n", " ", ""],  # Default separators prioritizing paragraphs
    )

documents = loader.load()
    # Or load page by page if you prefer
    # documents = loader.load()

#pages = loader.load_and_split()
split_documents = text_splitter.split_documents(documents)
classifier = pipeline("zero-shot-classification")
texts = [doc.page_content for doc in split_documents]

"""for text in texts:
   print(classifier(
    text,
    candidate_labels=["policies", "guildlines", "process"],
    )
   )"""



    # Load a pre-trained Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")

chroma_db = Chroma.from_texts(
    texts=documents,
    collection_name='db_docs',
    collection_metadata={"hnsw:space": "cosine"},  # Set distance function to cosine
embedding=embeddings
)



similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
search_kwargs={"k": 3,"score_threshold": 0.3})

query = "when should ICT carry out security testing? "
 
results = chroma_db.similarity_search(query=query, k=10)
print("**********************************")
print(query)
for doc in results:
      print(f"* {doc.page_content} [{doc.metadata}]")
     
