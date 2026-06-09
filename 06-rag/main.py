import os
from dotenv import load_dotenv
from operator import itemgetter
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(Path(__file__).parent / "../.env")

prompt_template = ChatPromptTemplate.from_template(
    """\
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    provide a detailed answer: 
    """)

embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING", ""))
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))

vector_store = PineconeVectorStore(
    index_name="langraph-udemy-eden-marco-rag",
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={'k': 3}
)

def format_docs(docs: dict) -> str:
    """Format retrieved documents into  single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LangChain expression language
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to cmpose with other chains
    - More verbose and error-prone
    """

    docs = retriever.invoke(query)
    context = format(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    return llm.invoke(messages)

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language)
    Returns a chain that can be invoked with {"question": ""}

    Advantages
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain-stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain ca be saved, shared and composed with other chains
    - Better debugging: LangChain provides better observability
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

def main(query: str):
    print('Initializing components')

    print("===" * 70)
    print("Implementation 0: Raw LLm without RAG")
    results = llm.invoke([HumanMessage(content=query)])
    print(f"Result: {results.content}")

    print("===" * 70)
    print("Implementation 1: Retrieval chain without LangChain expression language")
    results = retrieval_chain_without_lcel(query)
    print(f"Result: {results}")

    print("Implementation 2: Retrieval chain with LangChain expression language")
    chain = create_retrieval_chain_with_lcel()
    results = chain.invoke({"question": query})
    print(f"Result: {results}")


if __name__ == "__main__":
    query = "What is pinecone in machine learning?"

    main(query)
