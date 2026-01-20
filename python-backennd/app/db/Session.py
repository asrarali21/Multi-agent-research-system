
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os


DATABASE_URL = os.getenv("DATABASE_URL")


engine = create_engine(DATABASE_URL)


SessionLocal = sessionmaker(
    autoflush=False,
    bind=engine,
)

def get_db():
    """
    Creates a new database session for each request.
    Automatically closes the session after the request is done.
    """
    db = SessionLocal()
    try:
        yield db  
    finally:
        db.close()


