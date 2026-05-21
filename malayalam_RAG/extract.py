
import fitz

def extract_text_from_pdf(file_path):
    print(f"Opening...{file_path}")

    doc = fitz.open(file_path)
    print(len(doc))
    text = doc[1].get_text()
    print(text)

if __name__ == '__main__':
    test_pdf = 'kerala.pdf'

    try:
        extract_text_from_pdf(test_pdf)
    except Exception as e:
        print(e)
