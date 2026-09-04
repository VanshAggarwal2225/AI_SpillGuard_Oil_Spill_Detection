"""
FastAPI backend for AI SpillGuard - Oil Spill Detection
---------------------------------------------------------
Stores prediction records (image name, label, spill %, confidence,
threshold used, timestamp) in a local SQLite database using SQLModel.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST   /predictions        -> create a new prediction record
    GET    /predictions        -> list all prediction records (most recent first)
    GET    /predictions/{id}   -> get a single record
    DELETE /predictions/{id}   -> delete a record
    DELETE /predictions        -> clear all records
"""

from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select


# -------------------------------
# Database setup
# -------------------------------
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class PredictionRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    prediction: str          # "Oil Spill" or "No Oil Spill"
    spill_percentage: float
    confidence: float
    threshold: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionCreate(SQLModel):
    filename: str
    prediction: str
    spill_percentage: float
    confidence: float
    threshold: float


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="AI SpillGuard API", version="1.0.0")

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
def root():
    return {"status": "ok", "service": "AI SpillGuard API"}


@app.post("/predictions", response_model=PredictionRecord)
def create_prediction(record: PredictionCreate):
    db_record = PredictionRecord.model_validate(record)
    with Session(engine) as session:
        session.add(db_record)
        session.commit()
        session.refresh(db_record)
        return db_record


@app.get("/predictions", response_model=List[PredictionRecord])
def list_predictions():
    with Session(engine) as session:
        statement = select(PredictionRecord).order_by(PredictionRecord.id.desc())
        return session.exec(statement).all()


@app.get("/predictions/{record_id}", response_model=PredictionRecord)
def get_prediction(record_id: int):
    with Session(engine) as session:
        record = session.get(PredictionRecord, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        return record


@app.delete("/predictions/{record_id}")
def delete_prediction(record_id: int):
    with Session(engine) as session:
        record = session.get(PredictionRecord, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        session.delete(record)
        session.commit()
        return {"ok": True}


@app.delete("/predictions")
def clear_predictions():
    with Session(engine) as session:
        records = session.exec(select(PredictionRecord)).all()
        for r in records:
            session.delete(r)
        session.commit()
        return {"ok": True, "deleted": len(records)}
