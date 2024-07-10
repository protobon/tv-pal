from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import sys

load_dotenv()

loader = CSVLoader(file_path="./data/TV_show_data.csv", encoding="utf-8")
documents = loader.load()
db = Chroma.from_documents(documents, OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-3.5-turbo")

TEMPLATE = """
Answer questions about TV shows based exclusively on the following context:
{context}

Question: {question}
"""

if __name__ == "__main__":
    while True:
        query = input("query (q for quit): ")
        if query.lower().strip() == "q":
            sys.exit()
        docs = db.similarity_search(query, k=2)
        context = "\n".join(doc.page_content for doc in docs)
        prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
        prompt = prompt_template.format(context=context, question=query)
        response = llm.invoke(prompt)
        print(response.content)
