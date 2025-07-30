import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, String, ForeignKey, Integer, DateTime, Boolean, REAL, Text
from sqlalchemy.orm import relationship

from src.core.database.db_connection import Base

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    user_id = Column(String(50), nullable=False, index=True)
    organization_id = Column(String(50), nullable=True, index=True)
    api_key = Column(String(255), nullable=False, index=True, unique=True)
    name = Column(String(100), nullable=True)
    expiry_date = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    rate_limit = Column(Integer, default=100, nullable=False)  
    usage_count = Column(Integer, default=0, nullable=False)

class ChatSessions(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    title = Column(String(255), nullable=False)
    final_answer = Column(String(255), nullable=True)
    user_id = Column(String(50), nullable=False, index=True)
    organization_id = Column(String(50), nullable=True, index=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    state = Column(Integer, nullable=True)
    duration = Column(Integer, nullable=True)
    quantity_rating = Column(Integer, nullable=True)
    total_rating = Column(Integer, nullable=True)
    avg_rating = Column(Integer, nullable=True)

    messages = relationship("Messages", back_populates="session", cascade="all, delete-orphan")

class Documents(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    file_name = Column(String(255), nullable=False)
    collection_name = Column(String(255), nullable=False)
    extension = Column(String(255), nullable=False)
    size = Column(Integer, nullable=True)
    status = Column(Boolean, nullable=True)
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=True)
    sha256 = Column(String, nullable=False)
    organization_id = Column(String(50), nullable=True, index=True)
    file_url = Column(String, nullable=True)
    
    reference_docs = relationship("ReferenceDocs", back_populates="document")

class Messages(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    content = Column(String(10000), nullable=True)
    created_by = Column(String(255), nullable=True)
    question_id = Column(UUID(as_uuid=True), nullable=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)  
    updated_by = Column(String(255), nullable=True)
    type = Column(String(255), nullable=True)
    sender_role = Column(String(255), nullable=True)
    response_time = Column(REAL, nullable=True)
    organization_id = Column(String(50), nullable=True, index=True)

    session = relationship("ChatSessions", back_populates="messages")
    reference_docs = relationship("ReferenceDocs", back_populates="message")

class ReferenceDocs(Base):
    __tablename__ = "reference_docs"

    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id'), primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), primary_key=True)
    page = Column(Integer, nullable=True)

    message = relationship("Messages", back_populates="reference_docs")
    document = relationship("Documents", back_populates="reference_docs")

class Collection(Base):
    __tablename__ = "vectorstore_collection"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    user_id = Column(String(50), nullable=False, index=True)
    collection_name = Column(String(), nullable=True)
    organization_id = Column(String(50), nullable=True, index=True)
    is_personal = Column(Boolean, default=False, nullable=False)