
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

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




if __name__ == '__main__':
    test_pdf = 'kerala.pdf'

    try:
        result = extract_text_from_pdf(test_pdf)
        print(f'Total characters extracted: {len(result)}')
        chunks = chunk_text(result)
        print(f'Total chunks created: {len(chunks)}')
        print('Sample chunk:', chunks[0])

    except Exception as e:
        print(e)
