import uuid

from src.core.database.db_connection import db
from src.core.database.models.schemas import ChatSessions, Messages, ReferenceDocs, Documents
from src.core.utils.constants import MessageType
from src.core.utils.utils import extension_mapping
from src.core.utils.logger.custom_logging import LoggerMixin

class ChatRepository(LoggerMixin):
    """
    Repository class handling chat-related database operations using SQLAlchemy ORM.
    """
    def __init__(self):
        super().__init__()
        from src.core.database.repository.file_repository import FileProcessingRepository
        self._data_reprocessing_repository = FileProcessingRepository()

    def is_exist_session(self, session_id) -> bool:
        """
        Check if a chat session exists
        
        Args:
            session_id: The ID of the session to check
            
        Returns:
            bool: True if the session exists, False otherwise
        """
        try:
            with db.session_scope() as session:
                exists = session.query(session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).exists()).scalar()
                return exists
        except Exception as e:
            self.logger.error(f"Error checking if session exists: {str(e)}")
            raise ValueError(str(e))

    def save_user_question(self, session_id, created_at, created_by, content):
        """
        Save a user's question to the database
        
        Args:
            session_id: The ID of the chat session
            created_at: Timestamp when the question was created
            created_by: User who created the question
            content: The question text
            
        Returns:
            str: ID of the saved message
        """
        if not self.is_exist_session(session_id):
            self.logger.error("Chat session does not exist!")
            raise ValueError("Chat session does not exist")

        try:
            with db.session_scope() as session:
                message_id = str(uuid.uuid4())
                
                message = Messages(
                    id=message_id,
                    created_at=created_at,
                    created_by=created_by,
                    content=content,
                    type=MessageType.QUESTION,
                    session_id=session_id,
                    sender_role='user'
                )
                
                session.add(message)
                return message_id
        except Exception as e:
            self.logger.error(f"Error saving user question: {str(e)}")
            raise ValueError(str(e))

    def save_assistant_response(self, session_id, created_at, question_id, content, response_time):
        """
        Save the assistant's response
        
        Args:
            session_id: The ID of the chat session
            created_at: Timestamp when the response was created
            question_id: The ID of the question being answered
            content: The response text
            response_time: Time taken to generate the response
            
        Returns:
            str: ID of the saved message
        """
        if not self.is_exist_session(session_id):
            self.logger.error("Chat session does not exist!")
            raise ValueError("Chat session does not exist")

        try:
            with db.session_scope() as session:
                message_id = str(uuid.uuid4())
                
                message = Messages(
                    id=message_id,
                    created_at=created_at,
                    content=content,
                    type=MessageType.ANSWER,
                    question_id=question_id,
                    session_id=session_id,
                    sender_role='assistant',
                    response_time=response_time
                )
                
                session.add(message)
                return message_id
        except Exception as e:
            self.logger.error(f"Error saving assistant response: {str(e)}")
            raise ValueError(str(e))

    def update_assistant_response(self, updated_at, message_id, content, response_time):
        """
        Update an existing assistant response
        
        Args:
            updated_at: Timestamp of the update
            message_id: ID of the message to update
            content: Updated content
            response_time: Updated response time
        """
        try:
            with db.session_scope() as session:
                message = session.query(Messages).filter(Messages.id == message_id).first()
                if message:
                    message.updated_at = updated_at
                    message.content = content
                    message.response_time = response_time
        except Exception as e:
            self.logger.error(f"Error updating assistant response: {str(e)}")
            raise ValueError(str(e))

    def get_document_info_by_document_id(self, document_id):
        """
        Get document information by ID
        
        Args:
            document_id: The ID of the document
            
        Returns:
            dict: Document information
        """
        try:
            with db.session_scope() as session:
                document = session.query(Documents).filter(Documents.id == document_id).first()
                if document:
                    return {
                        'id': document.id,
                        'file_name': document.file_name,
                        'miniourl': document.miniourl
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting document info: {str(e)}")
            return None

    def get_chat_message_history_by_session_id(self, session_id, limit=5):
        """
        Get chat history for a session
        
        Args:
            session_id: The ID of the session
            limit: Maximum number of messages to retrieve
            
        Returns:
            List[Tuple]: List of (content, sender_role) tuples
        """
        try:
            with db.session_scope() as session:
                messages = session.query(Messages.content, Messages.sender_role)\
                    .filter(Messages.session_id == session_id)\
                    .order_by(Messages.created_at.desc())\
                    .limit(limit)\
                    .all()
                
                return messages
        except Exception as e:
            self.logger.error(f"Error getting chat history: {str(e)}")
            return []

    def get_sources_by_message_id(self, message_id):
        """
        Get sources referenced by a message
        
        Args:
            message_id: The ID of the message
            
        Returns:
            List[Dict]: List of source dictionaries
        """
        try:
            with db.session_scope() as session:
                # Join query to get reference documents and their document info
                results = session.query(
                    ReferenceDocs.message_id,
                    ReferenceDocs.document_id,
                    Documents.file_name,
                    Documents.miniourl,
                    ReferenceDocs.page
                ).join(
                    Documents,
                    ReferenceDocs.document_id == Documents.id
                ).filter(
                    ReferenceDocs.message_id == message_id
                ).order_by(
                    ReferenceDocs.document_id
                ).all()
                
                # Convert to dictionaries
                result_dict = [dict(zip(('message_id', 'document_id', 'file_name', 'miniourl', 'page'), r)) for r in results]
                
                # Add extension info
                for doc in result_dict:
                    file_extension = doc.get("file_name", "").split(".")[-1].lower()
                    doc["extension"] = extension_mapping.get(file_extension, file_extension)
                
                return result_dict
        except Exception as e:
            self.logger.error(f"Error getting sources by message: {str(e)}")
            return []

    def get_pageable_chat_history_by_session_id(self, session_id, page=1, size=10, sort='DESC'):
        """
        Get paginated chat history
        
        Args:
            session_id: The ID of the session
            page: Page number (1-based)
            size: Number of items per page
            sort: Sort order ('ASC' or 'DESC')
            
        Returns:
            List[Dict]: List of message dictionaries
        """
        if not self.is_exist_session(session_id):
            self.logger.error("Chat session does not exist!")
            raise ValueError("Chat session does not exist")

        try:
            with db.session_scope() as session:
                # Calculate offset
                offset = (page - 1) * size
                
                # Build query
                query = session.query(
                    Messages.id,
                    Messages.session_id,
                    Messages.content,
                    Messages.sender_role,
                    Messages.created_at
                ).filter(
                    Messages.session_id == session_id
                )
                
                # Apply sorting
                if sort.upper() == 'DESC':
                    query = query.order_by(Messages.created_at.desc())
                else:
                    query = query.order_by(Messages.created_at.asc())
                
                # Apply pagination
                query = query.offset(offset).limit(size)
                
                # Execute and format results
                results = query.all()
                return [dict(zip(('id', 'session_id', 'content', 'sender_role', 'created_at'), r)) for r in results]
        except Exception as e:
            self.logger.error(f"Error getting pageable chat history: {str(e)}")
            raise ValueError(str(e))

    def save_reference_docs(self, message_id, document_id, page):
        """
        Save reference document for a message
        
        Args:
            message_id: ID of the message
            document_id: ID of the document
            page: Page number in the document
        """
        try:
            with db.session_scope() as session:
                ref_doc = ReferenceDocs(
                    message_id=message_id,
                    document_id=document_id,
                    page=page
                )
                
                session.add(ref_doc)
        except Exception as e:
            self.logger.error(f"Error saving reference docs: {str(e)}")
            raise ValueError(str(e))

    def update_title_chat_session(self, session_id, title):
        """
        Update the title of a chat session
        
        Args:
            session_id: ID of the session
            title: New title
        """
        try:
            with db.session_scope() as session:
                chat_session = session.query(ChatSessions).filter(ChatSessions.id == session_id).first()
                if chat_session:
                    chat_session.title = title
        except Exception as e:
            self.logger.error(f"Error updating chat session title: {str(e)}")
            raise ValueError(str(e))

    def is_title_by_session_id(self, session_id):
        """
        Check if a session has a title
        
        Args:
            session_id: ID of the session
            
        Returns:
            bool: True if the session has a title, False otherwise
        """
        try:
            with db.session_scope() as session:
                title = session.query(ChatSessions.title).filter(ChatSessions.id == session_id).first()
                return title and title[0]
        except Exception as e:
            self.logger.error(f"Error checking session title: {str(e)}")
            raise ValueError(str(e))