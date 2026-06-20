import argparse

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are answering questions about Pride and Prejudice.

Answer ONLY using information found in the retrieved context.

Answer according to their relevance scores.

If the answer is explicitly stated in the context,
quote or paraphrase that statement directly.

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_text",
        type=str,
        help="The query text."
    )
    args = parser.parse_args()
    query_text = args.query_text

    # Use the same embedding model that was used to create the database
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Load Chroma database
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_function
    )

    # Search for relevant chunks
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    
    for doc, score in results:
        print("\n====================")
        print(f"Score: {score}")                                
        print(doc.page_content[:500])  

    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n".join(
    [
        f"PASSAGE {i+1} (relevance={score:.3f})\n{doc.page_content}"
        for i, (doc, score) in enumerate(results)
    ]
    )

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    print("\nSearching knowledge base...\n")

    # Local LLM
    model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
    )

    response = model.invoke(prompt)

    response_text = response.content

    sources = [
        doc.metadata.get("source", None)
        for doc, score in results
    ]

    print("\n================ ANSWER ================\n")
    print(response_text)

    print("\n================ SOURCES ================\n")
    for source in sources:
        print(source) 


if __name__ == "__main__":
    main()

