from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def search_database(query):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory="./kerala_db"
    )

    print(f'Searching for: {query}')
    results = db.similarity_search(query, k=3) #3 most relevant chunks

    return results

def get_answer(query, context_chunks):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) #temperature=0 for deterministic answers

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question ONLY using the provided context. Reply in Malayalam. If the answer is not in the context, say 'I cannot find the answer in the provided document.\n\nContext:\n{context}'"),
        ("user", "{question}")
    ])

    chain = prompt_template | llm
    response = chain.invoke({"question": query, "context": context_chunks})

    return response.content



if __name__ == '__main__':
    query = 'കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?'
    try:
        context_chunks = search_database(query)
        all_chunks = "\n\n".join([chunk.page_content for chunk in context_chunks])
        answer = get_answer(query, all_chunks)
        print(f'Answer: {answer}')

    except Exception as e:
        print(e)        