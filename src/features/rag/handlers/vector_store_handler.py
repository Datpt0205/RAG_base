from typing import Dict, Any, Optional, List
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.qdrant_connection_helper import QdrantConnection
from src.core.schemas.response import BasicResponse
from src.core.database.services.collection_management_service import CollectionManagementService


class VectorStoreQdrant(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.qdrant = QdrantConnection()
        self.collection_service = CollectionManagementService()


    def create_qdrant_collection(
        self, 
        collection_name: str, 
        user: Dict[str, Any],
        organization_id: Optional[str] = None,
        is_personal: bool = False
    ) -> BasicResponse:
        """
        Create a new collection in Qdrant and save metadata to PostgreSQL

        Args:
            collection_name: Name of the collection to create (use directly)
            user: Information about the user creating the collection
            organization_id: ID of the organization that owns the collection (optional)
            is_personal: True if it is a personal collection, False if it is an organization collection

        Returns:
            BasicResponse: Collection creation result
        """
        resp = BasicResponse(
            status="Success",
            message="create qdrant collection success.",
            data=collection_name
        )
        
        try:
            if not self.qdrant.client.collection_exists(collection_name=collection_name):
                # 1. Create collection in Qdrant vector database
                is_created = self.qdrant._create_collection(collection_name)
                
                if is_created:
                    # 2. Save metadata to PostgreSQL
                    try:
                        # Create record in vectorstore_collection table
                        collection_id = self.collection_service.create_collection(
                            collection_name=collection_name,
                            user_id=user["id"],
                            organization_id=None if is_personal else organization_id,
                            is_personal=is_personal
                        )
                        # self.logger.info(f"Collection metadata saved with ID: {collection_id}")

                        resp.message = f"create qdrant collection '{collection_name}' success."
                        resp.data = {
                            "collection_name": collection_name,
                            "organization_id": organization_id if not is_personal else None,
                            "is_personal": is_personal
                        }
                    except Exception as db_error:
                        # If saving to PostgreSQL fails, log it but still consider it successful because the collection was created in Qdrant
                        self.logger.error(f"Created Qdrant collection but failed to save metadata: {str(db_error)}")
                        resp.message = f"create qdrant collection '{collection_name}' success (metadata save failed)."
                else:
                    resp.message = f"create qdrant collection '{collection_name}' failed."
                    resp.status = "Failed"
                    resp.data = None
            else:
                resp.message = f"collection '{collection_name}' already exist."
                resp.status = "Failed"
                resp.data = None
            
            return resp
        except Exception as e:
            self.logger.error(f"create qdrant collection '{collection_name}' failed. Detail error: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Create qdrant collection {collection_name} failed. Detail error: {str(e)}",
                data=None
            )
        
    def delete_qdrant_collection(self, 
        collection_name: str, 
        user: Dict[str, Any],
        organization_id: Optional[str] = None,
        is_personal: bool = False
    ) -> BasicResponse:
        """
        Delete collection from Qdrant and PostgreSQL

        Args:
            collection_name: Collection name to delete
            user: User information
            organization_id: Organization ID that owns the collection (optional)
            is_personal: True if it is a personal collection, False if it is an organization collection

        Returns:
            BasicResponse: Collection deletion result
        """
        try:
            # 1. Check if collection exists
            if self.qdrant.client.collection_exists(collection_name=collection_name):
                #2. Check access before deleting
                has_permission = self.collection_service.check_collection_permission(
                    user_id=user["id"],
                    collection_name=collection_name,
                    organization_id=organization_id,
                    is_personal=is_personal,
                    required_permission="delete" 
                )
                
                if has_permission:
                    # 3. Delete collection from Qdrant
                    self.qdrant._delete_collection(collection_name)
                    
                    # 4. Delete metadata from PostgreSQL (id, user_id, collection_name, organization_id, is_personal)
                    try:
                        self.collection_service.delete_collection(
                            collection_name=collection_name,
                            organization_id=organization_id,
                            is_personal=is_personal
                        )
                    except Exception as db_error:
                        self.logger.error(f"Deleted Qdrant collection but failed to remove metadata: {str(db_error)}")
                    
                    return BasicResponse(
                        status="Success",
                        message=f"Delete qdrant collection '{collection_name}' success.",
                        data={"collection_name": collection_name, "is_personal": is_personal}
                    )
                else:
                    return BasicResponse(
                        status="Failed",
                        message=f"User is not authorized to delete {collection_name} collection",
                        data=None
                    )
            else:
                return BasicResponse(
                    status="Failed",
                    message=f"Collection {collection_name} is not exist.",
                    data=None
                )
        except Exception as e:
            self.logger.error(f"Delete qdrant collection '{collection_name}'failed. Detail error: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Delete qdrant collection '{collection_name}'failed. Detail error: {str(e)}",
                data=None
            )
    

    def list_qdrant_collections(self, 
        user: Dict[str, Any] = None,
        organization_id: Optional[str] = None,
        include_personal: bool = True,
        include_organizational: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get the list of collections from Qdrant

        Args:
            user: User information
            organization_id: Organization ID to filter the collection
            include_personal: Include personal collections
            include_organizational: Include organizational collections

        Returns:
            List[Dict[str, Any]]: List of collections with metadata
        """
        try:
            # 1. Get a list of all collections from Qdrant
            collections = self.qdrant.client.get_collections().collections
            all_collection_names = [c.name for c in collections]
            
            # 2. Get access information from PostgreSQL
            if user and "id" in user:
                user_collections = self.collection_service.get_user_collections(
                    user_id=user["id"],
                    organization_id=organization_id,
                    include_personal=include_personal,
                    include_organizational=include_organizational
                )
            else:
                user_collections = []
            
            # 3. Get name and information from PostgreSQL
            result_collections = []
            for collection in user_collections:
                collection_name = collection.get("collection_name")
                if collection_name in all_collection_names:
                    # Only return collections that actually exist in Qdrant
                    result_collections.append({
                        "collection_name": collection_name,
                        "is_personal": collection.get("is_personal", False),
                        "organization_id": collection.get("organization_id"),
                        "user_id": collection.get("user_id")
                    })
            
            return result_collections
        except Exception as e:
            self.logger.error(f"List qdrant collections failed. Detail error: {str(e)}")
            raise Exception(f"List qdrant collections failed: {str(e)}")


    

    # ======================= Functions for RankMate project =======================
    def create_qdrant_collection_store(self, collection_name: str) -> BasicResponse:
        """
        Create a new collection in Qdrant only (no PostgreSQL metadata)
        Simplified version for external use without user/organization tracking
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            BasicResponse: Collection creation result
        """
        resp = BasicResponse(
            status="Success",
            message="create qdrant collection success.",
            data=collection_name
        )
        
        try:
            if not self.qdrant.client.collection_exists(collection_name=collection_name):
                # Create collection in Qdrant vector database only
                is_created = self.qdrant._create_collection(collection_name)
                
                if is_created:
                    self.logger.info(f"Simple collection '{collection_name}' created successfully in Qdrant")
                    resp.message = f"create qdrant collection '{collection_name}' success."
                    resp.data = {
                        "collection_name": collection_name,
                        "type": "simple",
                        "metadata_stored": False
                    }
                else:
                    resp.message = f"create qdrant collection '{collection_name}' failed."
                    resp.status = "Failed"
                    resp.data = None
            else:
                resp.message = f"collection '{collection_name}' already exist."
                resp.status = "Failed"
                resp.data = None
            
            return resp
        except Exception as e:
            self.logger.error(f"create simple qdrant collection '{collection_name}' failed. Detail error: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Create qdrant collection {collection_name} failed. Detail error: {str(e)}",
                data=None
            )

    def delete_qdrant_collection_store(self, collection_name: str) -> BasicResponse:
        """
        Delete collection from Qdrant only (no PostgreSQL cleanup)
        Simplified version for external use
        
        Args:
            collection_name: Collection name to delete
            
        Returns:
            BasicResponse: Collection deletion result
        """
        try:
            # Check if collection exists
            if self.qdrant.client.collection_exists(collection_name=collection_name):
                # Delete collection from Qdrant directly
                self.qdrant._delete_collection(collection_name)
                
                self.logger.info(f"Simple collection '{collection_name}' deleted successfully from Qdrant")
                
                return BasicResponse(
                    status="Success",
                    message=f"Delete qdrant collection '{collection_name}' success.",
                    data={
                        "collection_name": collection_name,
                        "type": "simple",
                        "metadata_cleaned": False
                    }
                )
            else:
                return BasicResponse(
                    status="Failed",
                    message=f"Collection {collection_name} does not exist.",
                    data=None
                )
        except Exception as e:
            self.logger.error(f"Delete simple qdrant collection '{collection_name}' failed. Detail error: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Delete qdrant collection '{collection_name}' failed. Detail error: {str(e)}",
                data=None
            )

    def list_qdrant_collections_store(self) -> List[str]:
        """
        Get the list of all collections from Qdrant (simplified version)
        
        Returns:
            List[str]: List of collection names
        """
        try:
            # Get all collections from Qdrant
            collections = self.qdrant.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            self.logger.info(f"Found {len(collection_names)} collections in Qdrant")
            return collection_names
        except Exception as e:
            self.logger.error(f"List simple qdrant collections failed. Detail error: {str(e)}")
            raise Exception(f"List qdrant collections failed: {str(e)}")