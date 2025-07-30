import psycopg2
from urllib.parse import quote_plus as urlquote
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.core.utils.config import settings
from src.core.utils.config_loader import ConfigReaderInstance
from src.features.rag.helpers.singleton_helper import SingletonMeta
from src.core.utils.logger.custom_logging import LoggerMixin

# Read configuration from YAML file
db_config = ConfigReaderInstance.yaml.read_config_from_file(settings.DATABASE_CONFIG_FILENAME)
postgres_config = db_config.get('POSTGRES')

# URL connection to PostgreSQL
POSTGRES_CONNECTION_STRING = 'postgresql://{}:{}@{}:{}/{}'.format(
    postgres_config['USER'],
    urlquote(postgres_config['PASSWORD']),
    postgres_config['HOST'],
    postgres_config['PORT'],
    postgres_config['DATABASE_NAME']
)

# Base model for SQLAlchemy ORM
Base = declarative_base()

class DatabaseConnection(LoggerMixin, metaclass=SingletonMeta):
    """
    Singleton class manages database connections throughout the application.
    Use SQLAlchemy for ORM and psycopg2 directly for raw SQL when needed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.logger.info("Initialize database connection")
        
        # SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            POSTGRES_CONNECTION_STRING, 
            echo=postgres_config.get('SQLALCHEMY_ECHO', 'false').lower() == 'true',
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        
        # Session factory cho SQLAlchemy
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they do not exist
        Base.metadata.create_all(bind=self.engine)
        
        self.logger.info("Database connection initialized successfully")
    
    def get_session(self) -> Session:
        """Get a new SQLAlchemy session"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """Provides transaction scope for a sequence of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session database error: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_connection(self):
        """Get a direct psycopg2 connection to execute SQL"""
        return psycopg2.connect(POSTGRES_CONNECTION_STRING)
    
    @contextmanager
    def connection_scope(self):
        """Provides transaction scope for a sequence of operations with detailed logging."""
        connection = self.get_connection()
        self.logger.info("Connection obtained from pool")
        try:
            yield connection
            connection.commit()
            self.logger.info("Transaction committed successfully")
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Transaction rolled back due to error: {str(e)}")
            raise
        finally:
            self.logger.info("Closing connection")
            connection.close()

# Initialize a single instance of DatabaseConnection
db = DatabaseConnection()

# Dependency for FastAPI to get database session
def get_db():
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()

# Dependency function to get direct connection when needed
def get_connection():
    conn = db.get_connection()
    try:
        yield conn
    finally:
        conn.close()