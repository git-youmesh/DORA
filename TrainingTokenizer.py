"""Therefore, you don’t necessarily need to train your tokenizer on a very large
corpus; the corpus just needs to be representative of your domain and big
enough for the tokenizer to extract statistically significant measures.


#In a nutshell, the tokenizer is just trained to know
#which letter combinations are the most frequent in our corpus"""
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

import csv
from langchain_community.embeddings import HuggingFaceEmbeddings


length = 10

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

texts = [doc.page_content for doc in split_documents]

iter_dataset = iter(texts)

def tok_list(tokenizer, string):
 input_ids = tokenizer(string, add_special_tokens=False)["input_ids"]
 return [tokenizer.decode(tok) for tok in input_ids]

def batch_iterator(batch_size=10):
 for _ in tqdm(range(0, length, batch_size)):
    yield [next(iter_dataset)for _ in range(batch_size)]




tokenizer_T5 = AutoTokenizer.from_pretrained("google-t5/t5-base")
tokenizer_camembert = AutoTokenizer.from_pretrained("almanach/camembert-base")
Text ="""To ensure the correct implementation over time of ICT security policies, procedures,
protocols, and tools referred to in Title II, Chapter I of this Regulation, it is important
that financial entities correctly assign and maintain any roles and responsibilities
relating to ICT security, and that they lay down the consequences of non-compliance
with ICT security policies or procedures."""




print(f"Size of the vocabulary: {len(tokenizer_T5)}")
print(f"Size of the vocabulary: {len(tokenizer_camembert)}")




print(f'T5 tokens for "sex": {tok_list(tokenizer_T5,"Text")}')
print(f'CamemBERT tokens for "being":{tok_list(tokenizer_camembert,"Text")}')


print(tokenizer_T5.backend_tokenizer.normalizer)
print(tokenizer_camembert.backend_tokenizer.normalizer)

new_tokenizer =tokenizer_T5.train_new_from_iterator(batch_iterator(),vocab_size=12500)


tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1],reverse=False)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[257:280]]);





