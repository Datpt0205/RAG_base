import os
import asyncio
from src.core.database.repository.file_repository import FileProcessingRepository, FileProcessingVecDB
from src.core.schemas.response import BasicResponse
from src.core.utils.constants import TypeDatabase, TypeDocument
from src.features.rag.handlers.api_key_auth_handler import APIKeyAuth
from src.features.rag.handlers.data_ingestion_handler import DataIngestion
from src.features.rag.handlers.file_partition_handler import DocumentExtraction
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import Response, Depends, status
from fastapi.routing import APIRouter
from fastapi import UploadFile, Query, Body, File, Request, HTTPException, Header

from src.core.utils.config import settings


router = APIRouter(prefix="/document")

# API key authentication instance
api_key_auth = APIKeyAuth()

data_ingestion = DataIngestion()
document_extraction = DocumentExtraction()
file_management_dal = FileProcessingRepository()
file_vecdb = FileProcessingVecDB()


class DocumentSource(BaseModel):
    url: str 
    filename: Optional[str] = None

class DocumentSourceRequest(BaseModel):
    urls: List[DocumentSource]


# Management document (search, delete)
@router.get("/search", response_description="Search documents with various filters")
async def search_documents(
    response: Response,
    request: Request,
    keyword: Optional[str] = Query(None, description="Search by keyword in file name"),
    file_type: Optional[str] = Query(
        None, enum=TypeDocument.list(), description="Filter by document type: pdf/word/image/pptx"
    ),
    collection_name: Optional[str] = Query(None, description="Filter by collection name"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    created_after: Optional[str] = Query(None, description="Filter documents created after date (YYYY-MM-DD)"),
    created_before: Optional[str] = Query(None, description="Filter documents created before date (YYYY-MM-DD)"),
    limit: int = Query(10, description="Limit the number of results"),
    offset: int = Query(0, description="Skip records for pagination"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get organization_id from request state (set by API key auth)
        organization_id = getattr(request.state, "organization_id", None)
        
        # Map file_type to actual extension if specified
        extension = None
        if file_type:
            extension_map = {
                "pdf": "pdf",
                "word": "docx",
                "image": "image%",  # Use SQL LIKE wildcard
                "pptx": "pptx"
            }
            extension = extension_map.get(file_type)
        
        # Parse date strings to datetime objects if provided
        from datetime import datetime
        parsed_created_after = datetime.strptime(created_after, "%Y-%m-%d") if created_after else None
        parsed_created_before = datetime.strptime(created_before, "%Y-%m-%d") if created_before else None
        
        # Use FileManagementDAL to search files
        search_results = file_management_dal.search_files(
            keyword=keyword,
            extension=extension,
            collection_name=collection_name,
            created_by=created_by,
            created_after=parsed_created_after,
            created_before=parsed_created_before,
            organization_id=organization_id,
            limit=limit,
            offset=offset
        )
        
        if search_results and search_results.get("files"):
            response.status_code = status.HTTP_200_OK
            return BasicResponse(
                status="success",
                message=f"Found {search_results.get('total_count', 0)} documents matching your criteria",
                data=search_results
            )
        else:
            response.status_code = status.HTTP_404_NOT_FOUND
            return BasicResponse(
                status="failed",
                message="No documents found matching your search criteria",
                data={"total_count": 0, "files": []}
            )
            
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="error",
            message=f"An error occurred while searching documents: {str(e)}",
            data=None
        )

@router.delete("/delete", response_description="Delete document from PostgreSQL and vector database")
async def delete_document(
    document_id: str,
    response: Response,
    request: Request,
    type_db: str = Query(
        default=TypeDatabase.Qdrant.value,
        enum=TypeDatabase.list(),
        description="Select vector database type"
    ),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    organization_id = getattr(request.state, "organization_id", None)
    user_role = getattr(request.state, "role", None)
    
    try:
        # Get document information including collection_name
        document = file_management_dal.get_file_by_id(document_id)
        if not document:
            response.status_code = status.HTTP_404_NOT_FOUND
            return BasicResponse(
                status="failed",
                message=f"Document with ID {document_id} not found",
                data=None
            )
    
        if user_role != "ADMIN" and document.get("organization_id") != organization_id:
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="failed",
                message="You don't have permission to delete this document",
                data=None
            )
        
        # Get the necessary information from the document
        file_name = document.get("file_name")
        collection_name = document.get("collection_name")
        
        if not collection_name:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return BasicResponse(
                status="failed",
                message="Document doesn't have collection information",
                data=None
            )
        
        # 1. Delete in PostgreSQL
        deleted = file_management_dal.delete_file_record(document_id, organization_id)
        if not deleted:
            response.status_code = status.HTTP_404_NOT_FOUND
            return BasicResponse(
                status="failed",
                message="Failed to delete document from database",
                data=None
            )
        
        # 2. Delete in vector database
        vecdb_handler = FileProcessingVecDB()
        
        # Delete documents by ID
        await vecdb_handler.delete_document_by_batch_ids(
            document_ids=[document_id],
            type_db=type_db,
            collection_name=collection_name,
            organization_id=organization_id
        )
        
        # Delete documents by file name
        if file_name:
            await vecdb_handler.delete_document_by_file_name(
                file_name=file_name,
                type_db=type_db,
                collection_name=collection_name,
                organization_id=organization_id
            )
        
        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="success",
            message=f"Successfully deleted document: {file_name}",
            data=None
        )
        
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="error",
            message=f"Error deleting document: {str(e)}",
            data=None
        )


@router.post("/extract", response_description="Extract text from documents without storing in vector database")
async def extract_text(
    response: Response,
    request: Request,
    backend: str = Query("docling", description="Text extraction backend (pymupdf or docling)"),
    use_image_processing: bool = Query(True, description="Whether to process images with OCR"),
    use_image_description: bool = Query(False, description="Whether to generate descriptions for images using SmolVLM"),
    files: List[UploadFile] = File(..., description="Document files to process"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    async def process_file(file: UploadFile):
        try:
            file_data = await file.read()
            temp_file_path = data_ingestion._save_temp_file(file.filename, file_data)
            document_id = str(os.path.basename(temp_file_path))
            
            # Extract text with enhanced options
            result = await document_extraction.extract_text(
                file=file,
                backend=backend,
                temp_file_path=temp_file_path,
                document_id=document_id,
                use_image_processing=use_image_processing,
                use_image_description=use_image_description
            )
            
            if result.status == "success" and result.data is not None:
                serializable_docs = []
                for doc in result.data:
                    serializable_docs.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                return BasicResponse(
                    status=result.status,
                    message=result.message,
                    data=serializable_docs
                )
            
            return result
        except Exception as e:
            return BasicResponse(
                status="error",
                message=f"Failed to extract text from {file.filename}: {str(e)}",
                data=None
            )
        finally:
            await file.seek(0)

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    successful_results = [result for result in results if result.status == "success"]
    
    if successful_results:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
    
    return results


@router.post("/upload", response_description="Upload document, extract text, and store in vector database")
async def upload_document(
    response: Response,
    request: Request,
    collection_name: str = Query(..., description="Qdrant collection name to store the document"),
    backend: str = Query("docling", description="Text extraction backend (pymupdf or docling)"),
    use_image_processing: bool = Query(True, description="Whether to process images with OCR"),
    use_image_description: bool = Query(False, description="Whether to generate descriptions for images using SmolVLM"),
    document_data: DocumentSourceRequest = Body(..., description="List of document URLs to process"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    # Get organization_id from request state
    organization_id = getattr(request.state, "organization_id", None)
    user_id = getattr(request.state, "user_id", None)
        
    async def process_file(doc: DocumentSource):
        try:
            result = await data_ingestion.ingest(
                file_or_url=doc.url,
                collection_name=collection_name,
                backend=backend,
                organization_id=organization_id,
                user_id=user_id,
                filename=doc.filename,
                use_image_processing=use_image_processing,
                use_image_description=use_image_description
            )
            
            display_name = doc.filename or doc.url.split("/")[-1]

            return BasicResponse(
                status="success" if result["status"] == "success" else "failed",
                message=f"{'Successfully' if result['status'] == 'success' else 'Failed to'} process document from URL: {display_name}",
                data=result.get("data")
            )
        except Exception as e:
            display_name = doc.filename or doc.url.split("/")[-1]
            return BasicResponse(
                status="error",
                message=f"Failed to process document from URL {display_name}: {str(e)}",
                data=None
            )

    tasks = [process_file(doc) for doc in document_data.urls]
    results = await asyncio.gather(*tasks)

    successful_results = [result for result in results if result.status == "success"]
    
    if successful_results:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
    
    return results


@router.post("/upload_store", response_description="Upload document, extract text, and store in vector database")
async def upload_document_store(
    response: Response,
    request: Request,
    collection_name: str = Query(..., description="Qdrant collection name to store the document"),
    backend: str = Query("docling", description="Text extraction backend (pymupdf or docling)"),
    use_image_processing: bool = Query(True, description="Whether to process images with OCR"),
    use_image_description: bool = Query(False, description="Whether to generate descriptions for images using SmolVLM"),
    document_data: DocumentSourceRequest = Body(..., description="List of document URLs to process"),
):

    async def process_file(doc: DocumentSource):
        try:
            result = await data_ingestion.ingest_document_store(
                file_or_url=doc.url,
                collection_name=collection_name,
                backend=backend,
                filename=doc.filename,
                use_image_processing=use_image_processing,
                use_image_description=use_image_description
            )
            
            display_name = doc.filename or doc.url.split("/")[-1]

            return BasicResponse(
                status="success" if result["status"] == "success" else "failed",
                message=f"{'Successfully' if result['status'] == 'success' else 'Failed to'} process document from URL: {display_name}",
                data=result.get("data")
            )
        except Exception as e:
            display_name = doc.filename or doc.url.split("/")[-1]
            return BasicResponse(
                status="error",
                message=f"Failed to process document from URL {display_name}: {str(e)}",
                data=None
            )

    tasks = [process_file(doc) for doc in document_data.urls]
    results = await asyncio.gather(*tasks)

    successful_results = [result for result in results if result.status == "success"]
    
    if successful_results:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
    
    return results 