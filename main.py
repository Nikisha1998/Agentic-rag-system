from fastapi import FastAPI
from pydantic import BaseModel
from loader import load_and_index
from ask_agent import ask_question

app = FastAPI()

load_and_index("sample_doc/discharge_summary.pdf")

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(q: Query):
    answer = ask_question(q.query)
    return {"answer": answer}
