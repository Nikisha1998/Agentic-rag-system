from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

# Define a cleaner prompt template
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful medical assistant. Based on the following discharge summary, answer the question clearly and concisely.

    Context:
    {context}

    Question: {question}

    Answer in one sentence:
    """
)

def ask_question(query, persist_dir="db"):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    retriever = vectordb.as_retriever()
    # qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # Now create the RetrievalQA with custom prompt
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa.run(query)
