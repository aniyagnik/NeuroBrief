"""Database models and session helpers."""

from contextlib import contextmanager

from sqlalchemy import Column, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from neurobrief.config import DATABASE_URL

Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    summary = Column(Text)
    quiz = Column(Text)
    feedback = Column(Text)


class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True)
    video_filename = Column(Text)
    transcript_text = Column(Text)
    summary_text = Column(Text)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(36), primary_key=True)
    status = Column(String(20), nullable=False, default="pending")
    error = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    quiz = Column(Text, nullable=True)
    source_type = Column(String(20), nullable=True)
    source_detail = Column(Text, nullable=True)
    level = Column(String(20), nullable=True)
    stage = Column(String(40), nullable=True)


db_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
Base.metadata.create_all(db_engine)


def migrate_schema():
    insp = inspect(db_engine)
    if insp.has_table("jobs"):
        cols = {c["name"] for c in insp.get_columns("jobs")}
        if "stage" not in cols:
            with db_engine.begin() as conn:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN stage TEXT"))


migrate_schema()
SessionLocal = sessionmaker(bind=db_engine)


@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
