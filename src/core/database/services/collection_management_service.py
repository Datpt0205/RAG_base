import uuid
from typing import List, Optional, Dict, Any

from src.core.database.db_connection import db
from src.core.database.models.schemas import Collection
from src.core.utils.logger.custom_logging import LoggerMixin


class CollectionManagementService(LoggerMixin):
    """
    Service to manage collections metadata in the database.
    Synchronizes collection information between Qdrant and PostgreSQL.
    """

    def __init__(self):
        super().__init__()


    def create_collection(self, collection_name: str, user_id: str, organization_id: Optional[str] = None, is_personal: bool = False) -> str:
        """
        Create a new collection record in the database
        
        Args:
            collection_name: Name of the collection (direct name, no prefixes)
            user_id: ID of the user who owns this collection
            organization_id: ID of the organization that owns this collection
            is_personal: Whether this is a personal collection
            
        Returns:
            str: ID of the created collection record
        """
        try:
            with db.session_scope() as session:
                # Check if collection already exists
                query = session.query(Collection).filter_by(
                    collection_name=collection_name
                )
                
                # Add filters based on type
                if is_personal:
                    query = query.filter_by(is_personal=True)
                else:
                    query = query.filter_by(is_personal=False)
                    
                existing = query.first()
                
                if existing:
                    # self.logger.info(f"Collection {collection_name} already exists in database")
                    return str(existing.id)
                
                # Create new collection record
                collection_id = uuid.uuid4()
                new_collection = Collection(
                    id=collection_id,
                    user_id=user_id,
                    collection_name=collection_name,
                    organization_id=None if is_personal else organization_id,
                    is_personal=is_personal
                )
                
                session.add(new_collection)
                # Session is automatically committed by session_scope
                
                # self.logger.info(f"Created collection record for {collection_name} with ID {collection_id}")
                return str(collection_id)
                
        except Exception as e:
            self.logger.error(f"Error creating collection record: {str(e)}")
            raise


    def delete_collection(self, 
        collection_name: str,
        organization_id: Optional[str] = None,
        is_personal: bool = False
    ) -> bool:
        """
        Delete a collection record from the database
        
        Args:
            collection_name: Name of the collection to delete
            organization_id: ID of the organization that owns this collection
            is_personal: Whether this is a personal collection
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            with db.session_scope() as session:
                # Build delete query
                query = session.query(Collection).filter_by(
                    collection_name=collection_name
                )
                
                # Add filters based on type
                if is_personal:
                    query = query.filter_by(is_personal=True)
                elif organization_id:
                    query = query.filter_by(organization_id=organization_id, is_personal=False)
                
                # Delete collection record
                result = query.delete()
                
                # Session is automatically committed by session_scope
                if result > 0:
                    self.logger.info(f"Deleted collection record for {collection_name}")
                    return True
                else:
                    self.logger.warning(f"No collection record found for {collection_name}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error deleting collection record: {str(e)}")
            return False
    

    def check_collection_permission(self,
        user_id: str,
        collection_name: str,
        organization_id: Optional[str] = None,
        is_personal: bool = False,
        required_permission: str = "read"  # "read", "write", "delete"
    ) -> bool:
        """
        Check if a user has permission on a collection
        
        Args:
            user_id: ID of the user
            collection_name: Name of the collection
            organization_id: ID of the organization 
            is_personal: Whether this is a personal collection
            required_permission: Type of permission required
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        try:
            with db.session_scope() as session:
                # Get the collection
                query = session.query(Collection).filter_by(
                    collection_name=collection_name
                )
                
                collection = query.first()
                
                if not collection:
                    return False
                
                # Personal collections: only owner can access
                if collection.is_personal:
                    # For personal collections, only the owner can do anything
                    return collection.user_id == user_id
                
                # Organizational collections
                if collection.organization_id and collection.organization_id == organization_id:
                    # For org collections:
                    # - Everyone in org can read
                    # - Only owner and admins can write/delete
                    if required_permission == "read":
                        return True
                    else:
                        # Check if user is owner
                        if collection.user_id == user_id:
                            return True
                        
                        # Check if user is admin (would need user role service)
                        # This is simplified - in real code you'd check admin status
                        from src.features.rag.handlers.user_role_handler import UserRoleService
                        user_role_service = UserRoleService()
                        return user_role_service.is_admin(user_id, organization_id)
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking collection permission: {str(e)}")
            return False
        

    def get_user_collections(self, 
        user_id: str,
        organization_id: Optional[str] = None,
        include_personal: bool = True,
        include_organizational: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all collections belonging to a user, optionally filtered by organization
        
        Args:
            user_id: ID of the user
            organization_id: Optional ID of the organization to filter by
            include_personal: Whether to include personal collections
            include_organizational: Whether to include organizational collections
            
        Returns:
            List[Dict[str, Any]]: List of collection information
        """
        try:
            with db.session_scope() as session:
                collections = []
                
                # Get personal collections if requested
                if include_personal:
                    personal_query = session.query(Collection).filter_by(
                        user_id=user_id,
                        is_personal=True
                    )
                    
                    personal_collections = personal_query.all()
                    
                    for collection in personal_collections:
                        collections.append({
                            "id": str(collection.id),
                            "collection_name": collection.collection_name,
                            "is_personal": True,
                            "user_id": user_id
                        })
                
                # Get organizational collections if requested
                if include_organizational and organization_id:
                    org_query = session.query(Collection).filter_by(
                        organization_id=organization_id,
                        is_personal=False
                    )
                    
                    org_collections = org_query.all()
                    
                    for collection in org_collections:
                        collections.append({
                            "id": str(collection.id),
                            "collection_name": collection.collection_name,
                            "is_personal": False,
                            "organization_id": organization_id,
                            "user_id": collection.user_id
                        })
                
                return collections
                
        except Exception as e:
            self.logger.error(f"Error getting user collections: {str(e)}")
            return []


    def get_all_collections(
        self, 
        is_admin: bool = False,
        organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all collections in the database, optionally filtered by organization.
        Only admin users can get all collections.
        
        Args:
            is_admin: Whether the requesting user is an admin
            organization_id: Optional ID of the organization to filter by
            
        Returns:
            List[Dict[str, Any]]: List of all collection data
        """
        if not is_admin:
            return []
            
        try:
            with db.session_scope() as session:
                # Build query
                query = session.query(Collection)
                
                # Add organization filter if provided
                if organization_id:
                    query = query.filter_by(organization_id=organization_id)
                
                collections = query.all()
                
                # Convert to list of dictionaries with full metadata
                result = []
                for collection in collections:
                    result.append({
                        "id": str(collection.id),
                        "collection_name": collection.collection_name,
                        "user_id": collection.user_id,
                        "organization_id": collection.organization_id,
                        "is_personal": collection.is_personal
                    })
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting all collections: {str(e)}")
            return []