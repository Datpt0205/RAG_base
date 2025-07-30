import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.core.database.db_connection import db
from src.features.rag.helpers.qdrant_connection_helper import QdrantConnection
from src.core.utils.config import settings
from src.core.utils.constants import TypeDatabase
from src.core.utils.logger.custom_logging import LoggerMixin


class FileProcessingVecDB(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.qdrant_client = QdrantConnection()

    async def delete_document_by_file_name(self, 
                     file_name: str,
                     type_db: str = TypeDatabase.Qdrant.value,
                     collection_name: str = settings.QDRANT_COLLECTION_NAME,
                     organization_id: Optional[str] = None):
        if not file_name:
            self.logger.error('event=delete-document-by-file-name '
                            'message="Delete document by file name Failed. '
                            f'error="file_name is None. Please check your input again." ')
        else:
            self.logger.info('event=delete-document-in-vector-database message="Start delete ..."')
            
            if type_db == TypeDatabase.Qdrant.value:
                if not self.qdrant_client.client.collection_exists(collection_name=collection_name):
                    self.logger.warning(f"Collection {collection_name} does not exist, skipping delete operation")
                    return
                    
                await self.qdrant_client.delete_document_by_file_name(
                    document_name=file_name, 
                    collection_name=collection_name,
                    organization_id=organization_id
                )
    
    async def delete_document_by_batch_ids(self, 
                     document_ids: list[str],
                     type_db: str = TypeDatabase.Qdrant.value,
                     collection_name: str = settings.QDRANT_COLLECTION_NAME,
                     organization_id: Optional[str] = None):
        
        if not document_ids:
            self.logger.error('event=delete-document-by-batch-ids '
                            'message="Delete document by batch ids Failed. '
                            f'error="document_ids is None. Please check your input again." ')
        else: 
            self.logger.info('event=delete-document-in-vector-database message="Start delete ..."')
            
            if type_db == TypeDatabase.Qdrant.value:
                if not self.qdrant_client.client.collection_exists(collection_name=collection_name):
                    self.logger.warning(f"Collection {collection_name} does not exist, skipping delete operation")
                    return
                    
                await self.qdrant_client.delete_document_by_batch_ids(
                    document_ids=document_ids, 
                    collection_name=collection_name,
                    organization_id=organization_id
                )


class FileProcessingRepository(LoggerMixin):
    """
    Repository class for handling file operations in the database.
    Combines functionality from previous FileProcessingRepository and FileManagementDAL.
    """
    def __init__(self):
        super().__init__()

    def create_file_record(
        self,
        document_id: uuid,
        file_name: str,
        extension: str,
        file_url: str,
        created_by: str,
        size: int,
        sha256: str,
        collection_name: str,
        organization_id: Optional[str] = None,
    ) -> str:
        """
        Create a new file record in the database
        Returns the ID of the created record
        """
        # self.logger.info("Creating new file record")
        
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                sql = """
                    INSERT INTO documents (
                        id, file_name, collection_name, extension, size, 
                        status, created_by, created_at, sha256, organization_id, file_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """
                created_at = datetime.now()
                status = True  # Active status
                
                cursor.execute(
                    sql, 
                    (
                        document_id, file_name, collection_name, extension, size,
                        status, created_by, created_at, sha256, organization_id, file_url
                    )
                )
                
                result = cursor.fetchone()
                returned_id = result[0] if result else document_id
                # self.logger.info(f"returned_id {returned_id}")
                # self.logger.info(f"Created file record for {file_name} with ID {returned_id}")
                return returned_id
                
        except Exception as e:
            self.logger.error(f"Error creating file record: {str(e)}")
            raise

    def get_file_by_id(self, document_id: str, organization_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get file information by ID
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["id = %s"]
                params = [document_id]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"""
                    SELECT id, file_name, collection_name, extension, size, 
                           status, created_by, created_at, sha256, organization_id, file_url
                    FROM documents
                    WHERE {where_clause}
                """
                
                cursor.execute(sql, params)
                result = cursor.fetchone()
                
                if not result:
                    return None
                    
                return {
                    "id": result[0],
                    "file_name": result[1],
                    "collection_name": result[2],
                    "extension": result[3],
                    "size": result[4],
                    "status": result[5],
                    "created_by": result[6],
                    "created_at": result[7],
                    "sha256": result[8],
                    "organization_id": result[9],
                    "file_url": result[10] if len(result) > 10 else None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting file by ID {document_id}: {str(e)}")
            return None

    def get_files_by_collection(self, collection_name: str, organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all files belonging to a collection
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["collection_name = %s", "status = TRUE"]
                params = [collection_name]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"""
                    SELECT id, file_name, collection_name, extension, size, 
                        created_at, created_by, sha256, organization_id, file_url
                    FROM documents
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                """
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                files = []
                for result in results:
                    files.append({
                        "id": result[0],
                        "file_name": result[1],
                        "collection_name": result[2],
                        "extension": result[3],
                        "size": result[4],
                        "created_at": result[5],
                        "created_by": result[6],
                        "sha256": result[7],
                        "organization_id": result[8],
                        "file_url": result[9]
                    })
                    
                return files
                
        except Exception as e:
            self.logger.error(f"Error getting files for collection {collection_name}: {str(e)}")
            raise

    def update_file_record(self, document_id: str, updates: Dict[str, Any], organization_id: Optional[str] = None) -> bool:
        """
        Update an existing file record
        Returns True if successful, False otherwise
        """
        if not updates:
            return False
            
        try:
            with db.connection_scope() as connection:  
                cursor = connection.cursor()
                set_clause = ", ".join([f"{key} = %s" for key in updates.keys()])
                values = list(updates.values())
                
                conditions = ["id = %s"]
                where_params = [document_id]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    where_params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"""
                    UPDATE documents
                    SET {set_clause}
                    WHERE {where_clause}
                """
                
                cursor.execute(sql, values + where_params)
                
                rows_affected = cursor.rowcount
                return rows_affected > 0
                
        except Exception as e:
            self.logger.error(f"Error updating file record {document_id}: {str(e)}")
            raise

    def delete_file_record(self, document_id: str, organization_id: Optional[str] = None) -> bool:
        """
        Delete a file record by ID
        Returns True if successful, False otherwise
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                # 1. First, delete all references to this document in the reference_docs table
                ref_sql = "DELETE FROM reference_docs WHERE document_id = %s"
                cursor.execute(ref_sql, [document_id])
                # self.logger.info(f"Deleted {cursor.rowcount} references to document {document_id}")
                
                # 2. Delete document
                conditions = ["id = %s"]
                params = [document_id]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"DELETE FROM documents WHERE {where_clause}"
                cursor.execute(sql, params)
                
                rows_affected = cursor.rowcount
                return rows_affected > 0
                
        except Exception as e:
            self.logger.error(f"Error deleting file record {document_id}: {str(e)}")
            raise

    def delete_record_by_collection(self, collection_name: str, organization_id: Optional[str] = None) -> int:
        """
        Delete all file records belonging to a collection
        Returns the number of records deleted
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                # 1. Get all document IDs in this collection
                doc_conditions = ["collection_name = %s"]
                doc_params = [collection_name]
                
                if organization_id:
                    doc_conditions.append("organization_id = %s")
                    doc_params.append(organization_id)
                    
                doc_where_clause = " AND ".join(doc_conditions)
                
                doc_sql = f"SELECT id FROM documents WHERE {doc_where_clause}"
                cursor.execute(doc_sql, doc_params)
                document_ids = [row[0] for row in cursor.fetchall()]
                
                total_deleted = 0
                
                if document_ids:
                    # 2. Remove all references in reference_docs for these documents
                    doc_ids_placeholders = ','.join(['%s'] * len(document_ids))
                    ref_sql = f"DELETE FROM reference_docs WHERE document_id IN ({doc_ids_placeholders})"
                    cursor.execute(ref_sql, document_ids)
                    ref_deleted = cursor.rowcount
                    # self.logger.info(f"Deleted {ref_deleted} references for {len(document_ids)} documents in collection {collection_name}")
                
                # 3. Delete all documents belonging to the collection
                conditions = ["collection_name = %s"]
                params = [collection_name]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"DELETE FROM documents WHERE {where_clause}"
                cursor.execute(sql, params)
                
                docs_deleted = cursor.rowcount
                total_deleted = docs_deleted
                
                # self.logger.info(f"Deleted {docs_deleted} documents from collection {collection_name}")
                return total_deleted
                
        except Exception as e:
            self.logger.error(f"Error deleting records for collection {collection_name}: {str(e)}")
            raise

    def get_file_count_by_collection(self, collection_name: str, organization_id: Optional[str] = None) -> int:
        """
        Get the number of files in a collection
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["collection_name = %s", "status = TRUE"]
                params = [collection_name]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
                cursor.execute(sql, params)
                result = cursor.fetchone()
                
                return result[0] if result else 0
                
        except Exception as e:
            self.logger.error(f"Error getting file count for collection {collection_name}: {str(e)}")
            raise

    def check_file_exists(self, file_name: str, sha256: str, organization_id: Optional[str] = None) -> bool:
        """
        Check if a file with the given name and hash exists
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["file_name = %s", "sha256 = %s"]
                params = [file_name, sha256]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"SELECT id FROM documents WHERE {where_clause} LIMIT 1"
                cursor.execute(sql, params)
                result = cursor.fetchone()
                
                return result is not None
                
        except Exception as e:
            self.logger.error(f"Error checking if file exists {file_name}: {str(e)}")
            raise

    def get_file_metadata(self, document_id: str, organization_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Get file metadata (name, collection_name) by ID
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["id = %s"]
                params = [document_id]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                sql = f"SELECT file_name, collection_name FROM documents WHERE {where_clause} LIMIT 1"
                cursor.execute(sql, params)
                result = cursor.fetchone()
                
                if not result:
                    return None, None
                    
                return result[0], result[1]
                
        except Exception as e:
            self.logger.error(f"Error getting file metadata for ID {document_id}: {str(e)}")
            raise
            
    def search_files(
        self, 
        keyword: Optional[str] = None,
        extension: Optional[str] = None,
        collection_name: Optional[str] = None,
        created_by: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        organization_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search for files based on various criteria
        """
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                conditions = ["status = TRUE"]
                params = []
                
                if keyword:
                    conditions.append("LOWER(file_name) LIKE %s")
                    params.append(f"%{keyword.lower()}%")
                    
                if extension:
                    conditions.append("extension = %s")
                    params.append(extension)
                    
                if collection_name:
                    conditions.append("collection_name = %s")
                    params.append(collection_name)
                    
                if created_by:
                    conditions.append("created_by = %s")
                    params.append(created_by)
                    
                if created_after:
                    conditions.append("created_at >= %s")
                    params.append(created_after)
                    
                if created_before:
                    conditions.append("created_at <= %s")
                    params.append(created_before)
                    
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                    
                where_clause = " AND ".join(conditions)
                
                # Count query
                count_sql = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
                cursor.execute(count_sql, params)
                total_count = cursor.fetchone()[0]
                
                # Data query
                data_sql = f"""
                    SELECT id, file_name, collection_name, extension, size, 
                        created_at, created_by, sha256, organization_id
                    FROM documents 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                cursor.execute(data_sql, params + [limit, offset])
                results = cursor.fetchall()
                
                files = []
                for result in results:
                    files.append({
                        "id": result[0],
                        "file_name": result[1],
                        "collection_name": result[2],
                        "extension": result[3],
                        "size": result[4],
                        "created_at": result[5],
                        "created_by": result[6],
                        "sha256": result[7],
                        "organization_id": result[8]
                    })
                    
                return {
                    "total_count": total_count,
                    "files": files
                }
                
        except Exception as e:
            self.logger.error(f"Error searching files: {str(e)}")
            raise


    def create_file_records(self, file_name, extension, file_url, uploaded_by, size, sha256, collection_name='', organization_id=None):
        return self.create_file_record(
            document_id=str(uuid.uuid4()),
            file_name=file_name,
            extension=extension,
            file_url=file_url,
            created_by=uploaded_by,
            size=size,
            sha256=sha256,
            collection_name=collection_name,
            organization_id=organization_id
        )

    def check_duplicates(self, sha256, file_name, organization_id=None):
        return self.check_file_exists(file_name, sha256, organization_id)

    def get_files_by_search_engine(self, key_word=None, extension=None, created_at=None, 
                                  limit=10, offset=0, organization_id=None):
        # Convert to using method search_files
        return self.search_files(
            keyword=key_word,
            extension=extension,
            created_after=created_at,
            limit=limit,
            offset=offset,
            organization_id=organization_id
        )

    def delete_document_by_batch_ids(self, document_ids: list[str], organization_id=None):
        try:
            if not document_ids:
                return None
                
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                
                for doc_id in document_ids:
                    # Delete reference docs
                    ref_sql = "DELETE FROM reference_docs WHERE document_id = %s"
                    cursor.execute(ref_sql, [doc_id])
                
                # Delete all documents
                ids_list = ', '.join(['%s'] * len(document_ids))
                
                if organization_id:
                    sql = f"DELETE FROM documents WHERE id IN ({ids_list}) AND organization_id = %s"
                    params = document_ids + [organization_id]
                else:
                    sql = f"DELETE FROM documents WHERE id IN ({ids_list})"
                    params = document_ids
                    
                cursor.execute(sql, tuple(params))
                
                return None
        except Exception as e:
            self.logger.error(f"Error deleting documents by batch IDs: {str(e)}")
            raise

    def delete_document_by_file_name(self, file_name, organization_id=None):
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                
                conditions = ["file_name = %s"]
                params = [file_name]
                
                if organization_id:
                    conditions.append("organization_id = %s")
                    params.append(organization_id)
                
                where_clause = " AND ".join(conditions)
                
                # Get all document_ids
                id_sql = f"SELECT id FROM documents WHERE {where_clause}"
                cursor.execute(id_sql, params)
                doc_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete reference docs
                if doc_ids:
                    for doc_id in doc_ids:
                        ref_sql = "DELETE FROM reference_docs WHERE document_id = %s"
                        cursor.execute(ref_sql, [doc_id])
                
                # Delete documents
                sql = f"DELETE FROM documents WHERE {where_clause}"
                cursor.execute(sql, params)
                
                return None
        except Exception as e:
            self.logger.error(f"Error deleting document by file name: {str(e)}")
            raise

    def get_document_by_id(self, document_id, organization_id=None):
        # Convert to get_file_metadata
        result = self.get_file_metadata(document_id, organization_id)
        return result

    def get_file_details_by_id(self, document_id) -> Optional[Dict[str, Any]]:
        # Convert to get_file_by_id
        return self.get_file_by_id(document_id)

    def get_file_details_by_name(self, file_name) -> Optional[Dict[str, Any]]:
        # Find files by name
        try:
            with db.connection_scope() as connection:
                cursor = connection.cursor()
                sql = """
                    SELECT id, file_name, collection_name, extension, size, status, 
                           created_by, created_at, sha256, organization_id, file_url
                    FROM documents 
                    WHERE file_name = %s 
                    LIMIT 1
                """
                cursor.execute(sql, (file_name,))
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                return {
                    'id': result[0],
                    'file_name': result[1],
                    'collection_name': result[2],
                    'extension': result[3],
                    'size': result[4],
                    'status': result[5],
                    'created_by': result[6],
                    'created_at': result[7],
                    'sha256': result[8],
                    'organization_id': result[9],
                    'file_url': result[10] if len(result) > 10 else None
                }
        except Exception as e:
            self.logger.error(f"Error getting file details by name: {str(e)}")
            return None

    def get_files_by_organization(self, organization_id, limit=100, offset=0) -> List[Dict[str, Any]]:
        # Use search_files with organization_id
        result = self.search_files(
            organization_id=organization_id,
            limit=limit,
            offset=offset
        )
        return result.get('files', [])