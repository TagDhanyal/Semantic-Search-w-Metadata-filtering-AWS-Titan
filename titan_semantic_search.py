# Import necessary libraries
from urllib.request import urlretrieve
import glob
from pypdf import PdfReader, PdfWriter
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

# Define the URLs of the PDF documents to retrieve
urls = [
    'https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/2022-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/2021-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/ar/Amazon-2020-Shareholder-Letter-and-1997-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/ar/2019-Shareholder-Letter.pdf'
]

# Define the filenames for the retrieved PDF documents
filenames = [
    'AMZN-2022-Shareholder-Letter.pdf',
    'AMZN-2021-Shareholder-Letter.pdf',
    'AMZN-2020-Shareholder-Letter.pdf',
    'AMZN-2019-Shareholder-Letter.pdf'
]

# Define the metadata for each PDF document
metadata = [
    dict(year=2022, source=filenames[0]),
    dict(year=2021, source=filenames[1]),
    dict(year=2020, source=filenames[2]),
    dict(year=2019, source=filenames[3])]

# Define the data root directory where the retrieved PDF documents will be saved
# Define the absolute path to the 'data' directory
data_root = os.path.abspath("data")

# Check if the 'data' directory exists, and create it if it doesn't
if not os.path.exists(data_root):
    os.makedirs(data_root)

# Retrieve the PDF documents from the URLs and save them to the data root directory
for idx, url in enumerate(urls):
    file_path = os.path.join(data_root, filenames[idx])  # Use os.path.join to create an absolute path
    urlretrieve(url, file_path)

# Preprocess the PDF documents by removing the first three pages and saving the remaining pages to a new PDF file
local_pdfs = glob.glob(data_root + '*.pdf')

documents = []

for idx, file in enumerate(filenames):
    loader = PyPDFLoader(os.path.join(data_root, file))
    document = loader.load()
    for document_fragment in document:
        document_fragment.metadata = metadata[idx]
        
    print(f'{len(document)} {document}\n')
    documents += document

# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(documents)
# Import necessary libraries
import os
import boto3
import json
import sys

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=True
)

from utils.TokenCounterHandler import TokenCounterHandler
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock

# Initialize Token Counter Handler
token_counter = TokenCounterHandler()

# Initialize BedrockEmbeddings
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Define your document list here

db = FAISS.from_documents(docs, embeddings)

# Save and load the vector store from the local filesystem
db.save_local("faiss_titan_index")
new_db = FAISS.load_local("faiss_titan_index", embeddings)
db = new_db

# Set your search query
query = "How has AWS evolved?"

# Basic Similarity Search
results_with_scores = db.similarity_search_with_score(query)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\nScore: {score}\n\n")

# Similarity Search with Metadata Filtering
filter = dict(year=2022)
results_with_scores = db.similarity_search_with_score(query, filter=filter)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}, Score: {score}\n\n")

# Top-K Matching
k = 2
fetch_k = 4
results = db.similarity_search(query, filter=filter, k=k, fetch_k=fetch_k)
for doc in results:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n\n")

# Maximal Marginal Relevance
results = db.max_marginal_relevance_search(query, filter=filter)
for doc in results:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n\n")

# Set up LangChain LLM with specified parameters
model_kwargs_titan = {
    "maxTokenCount": 512,
    "stopSequences": [],
    "temperature": 0.0,
    "topP": 0.5
}

# Initialize LLM
llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    client=bedrock_client,
    model_kwargs=model_kwargs_titan,
    callbacks=[token_counter]
)

# Test LLM with a prompt
prompt = "How has AWS evolved?"
result = llm.predict(prompt)
print(result)
token_counter.report()

# Define a prompt template
prompt_template = """
Human: Here is a set of context, contained in <context> tags:

<context>
{context}
</context>

Use the context to provide an answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Initialize RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": filter},
        callbacks=[token_counter]
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT
},
    callbacks=[token_counter]
)

# Test the RetrievalQA chain with queries
queries = [
    "How has AWS evolved?",
    "Why is Amazon successful?",
    "What business challenges has Amazon experienced?",
    "How was Amazon impacted by COVID-19?",
]

for query in queries:
    result = qa({"query": query})
    print(f'Query: {result["query"]}\n')
    print(f'Result: {result["result"]}\n')
    print(f'Context Documents: ')
    for srcdoc in result["source_documents"]:
        print(f'{srcdoc}\n')