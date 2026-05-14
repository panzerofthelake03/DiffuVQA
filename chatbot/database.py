from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///chatbot/chat_history.db", echo=False)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String(512))
    question   = Column(Text)
    answer     = Column(Text)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def save(image_path: str, question: str, answer: str):
    session = Session()
    try:
        record = ChatHistory(
            image_path=image_path,
            question=question,
            answer=answer,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def get_recent(limit: int = 20):
    session = Session()
    try:
        records = (
            session.query(ChatHistory)
            .order_by(ChatHistory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "question": r.question,
                "answer": r.answer,
                "image_path": r.image_path,
            }
            for r in records
        ]
    finally:
        session.close()


def get_all_count():
    session = Session()
    try:
        return session.query(ChatHistory).count()
    finally:
        session.close()
