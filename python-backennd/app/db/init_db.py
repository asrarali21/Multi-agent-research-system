from app.db.base import Base
from app.db.session import engine





def init_db():
    """Initialize database tables"""
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created!")