
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def extract_text_from_pdf(file_path):
    print(f"Opening...{file_path}")

    extracted_text = ""

    doc = fitz.open(file_path)
    for page_no in range(len(doc)):
        page = doc[page_no]
        extracted_text += page.get_text()

    return extracted_text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        length_function=len
        )
    
    chunks = text_splitter.split_text(text)
    return chunks

def vector_db(chunks):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Creating vector db...")
    db = Chroma.from_texts(
        texts = chunks,
        embedding = embedding_model,
        persist_directory="./kerala_db"
    )
    return db

    

if __name__ == '__main__':
    test_pdf = 'kerala.pdf'

    try:
        #extract
        result = extract_text_from_pdf(test_pdf)
        print(f'Total characters extracted: {len(result)}')

        #chunk
        chunks = chunk_text(result)
        print(f'Total chunks created: {len(chunks)}')
        print('Sample chunk:', chunks[0])

        #vector_db
        vector_database = vector_db(chunks)
        total_vectors = vector_database._collection.count()
        print(f'Total vectors in the database: {total_vectors}')
        
    except Exception as e:
        print(e)
