import uuid
import re
from fastapi import UploadFile
from pathlib import Path
import numpy as np
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from transformers import AutoModelForVision2Seq
_orig_from_pretrained = AutoModelForVision2Seq.from_pretrained

def _patched_from_pretrained(*args, **kwargs):
    kwargs["_attn_implementation"] = "eager"
    return _orig_from_pretrained(*args, **kwargs)
AutoModelForVision2Seq.from_pretrained = _patched_from_pretrained

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, FormatToExtensions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from langchain_core.documents import Document
import pymupdf, pymupdf4llm

# Import necessary options for image processing
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    smolvlm_picture_description
)
from docling_core.types.doc import ImageRefMode, PictureItem
from rapidocr_onnxruntime import RapidOCR

from src.core.schemas.response import BasicResponse
from src.core.utils.logger.custom_logging import LoggerMixin

# Get supported file formats from docling
SUPPORTED_FORMATS = [item for sublist in FormatToExtensions.values() for item in sublist]


class DocumentExtraction(LoggerMixin):
    _instance = None
    @classmethod
    def get_instance(cls):  
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        
        # Initialize RapidOCR at startup with multilingual support
        self.ocr_engine = RapidOCR(lang="en,vi,zh")
        
        # Configure default PDF processing options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.ocr_options.lang = ["es"]
        
        # Initialize enhanced pipeline options for image extraction
        self.enhanced_pipeline_options = PdfPipelineOptions()
        self.enhanced_pipeline_options.images_scale = 1.0
        self.enhanced_pipeline_options.generate_page_images = True
        self.enhanced_pipeline_options.generate_picture_images = True
        self.enhanced_pipeline_options.do_ocr = False  # We'll do OCR ourselves with RapidOCR
        self.enhanced_pipeline_options.do_table_structure = True
        self.enhanced_pipeline_options.table_structure_options.do_cell_matching = True
        self.enhanced_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )

    async def extract_text(self,
                    backend: str,
                    file: UploadFile, 
                    temp_file_path: str, 
                    document_id: uuid,
                    use_image_processing: bool = True,
                    use_image_description: bool = False):
        
        valid = self.validate_file_extension(file)
        # print(f"File extension: {valid}")
        # Extract text using the appropriate backend
        if valid == "pdf":
            if backend == "docling" and use_image_processing:
                # Use OCR-enhanced extraction
                self.logger.info(f"Using OCR-enhanced extraction for {file.filename}")
                markdown_text = await self.enhanced_ocr_extract(temp_file_path, use_image_description=use_image_description)
            elif backend == "docling":
                # Use regular docling extraction
                self.logger.info(f"Using regular docling extraction for {file.filename}")
                markdown_text = await self.docling_extract(temp_file_path)
            else:
                # Use pymupdf extraction
                self.logger.info(f"Using pymupdf extraction for {file.filename}")
                markdown_text = await self.pymupdf_extract(temp_file_path)
        else:
            # Use docling for other file types
            self.logger.info(f"Using docling extraction for non-PDF file: {file.filename}")
            markdown_text = await self.docling_extract(temp_file_path)
        
        # AFTER getting full markdown content with OCR, proceed with chunking
        self.logger.info(f"Now proceeding with chunking for {file.filename}")
        
        # Define headers for splitting
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]

        # Split based on markdown headers
        try:
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
            md_splits = md_splitter.split_text(markdown_text)
            # self.logger.info(f"Split document into {len(md_splits)} header sections")
            
            # Set chunking parameters
            chunk_size = 300
            chunk_overlap = 0
            
            # Split each header section into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            documents = text_splitter.split_documents(md_splits)
            
            # If header splitting didn't produce multiple chunks, try character splitting
            if len(documents) <= 1:
                # self.logger.info(f"Header splitting produced only one chunk, trying character splitting")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                documents = text_splitter.create_documents(
                    [markdown_text], 
                    metadatas=[{
                        'document_name': file.filename,
                        'headers': file.filename,
                        'document_id': document_id
                    }]
                )
        except Exception as e:
            # In case of chunking error, fall back to character splitting
            self.logger.error(f"Error during header chunking: {str(e)}, falling back to character splitting")
            chunk_size = 300
            chunk_overlap = 0
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            documents = text_splitter.create_documents(
                [markdown_text], 
                metadatas=[{
                    'document_name': file.filename,
                    'headers': file.filename,
                    'document_id': document_id
                }]
            )

        # Add metadata to each chunk
        for idx, document in enumerate(documents):
            headers = document.metadata.copy() if hasattr(document, 'metadata') else {}
            headers_string = ', '.join([f"{key}: {value}" for key, value in headers.items() if key not in ['document_name', 'document_id']])

            document.metadata = {
                'document_name': file.filename,
                'index': idx,
                'headers': headers_string or file.filename,
                'document_id': document_id
            }
        
        # self.logger.info(f"Created {len(documents)} chunks for {file.filename}")
        
        # Create response
        if len(documents) > 0:
            response = BasicResponse(status='success',
                                     message='Extract text from file successfully.',
                                     data=documents)
        else:
            response = BasicResponse(status='Failed',
                                    message='Extract text from file was failed.',
                                    data=None)
        return response
    
    async def pymupdf_extract(self, temp_file_path: str):
        """Extract text using PyMuPDF"""
        doc = pymupdf.open(temp_file_path)
        header_identifier = pymupdf4llm.IdentifyHeaders(doc, body_limit=6)
        markdown_text = pymupdf4llm.to_markdown(
                doc, 
                hdr_info=header_identifier.get_header_id, 
            )
        return markdown_text
    
    async def docling_extract(self, temp_file_path: str):
        """Extract text using regular docling extraction"""
        doc_converter = (DocumentConverter(allowed_formats=[
                        InputFormat.PDF,
                        InputFormat.IMAGE,
                        InputFormat.DOCX,
                        InputFormat.HTML,
                        InputFormat.PPTX,
                        InputFormat.ASCIIDOC,
                        InputFormat.MD,
                    ], 
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipeline_options
                    ),
                },
            )
        )
        result = doc_converter.convert(temp_file_path)
        markdown_text = result.document.export_to_markdown()
        return markdown_text 
    
    async def enhanced_ocr_extract(self, temp_file_path: str, use_image_description: bool = False):
        """
        Enhanced extraction with OCR for images and optional image description with SmolVLM
        """
        try:
            self.enhanced_pipeline_options.do_picture_description = use_image_description
            if use_image_description:
                self.enhanced_pipeline_options.picture_description_options = smolvlm_picture_description
                self.logger.info("Image description with SmolVLM enabled")
            
            # Initialize document converter with enhanced options
            doc_converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.enhanced_pipeline_options
                    )
                }
            )
            
            # Convert document
            # self.logger.info(f"Converting document using enhanced options: {temp_file_path}")
            result = doc_converter.convert(temp_file_path)
            document = result.document
            
            # Get document filename for logging
            doc_filename = Path(temp_file_path).stem
            
            # Process images
            image_content_data = []
            picture_counter = 0
            
            # Collect data from document
            for element, _level in document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    try:
                        # Get image
                        pil_image = element.get_image(document)
                        
                        if pil_image:
                            self.logger.info(f"Processing image #{picture_counter}")
                            
                            # Get OCR text
                            ocr_text = self._perform_ocr_in_memory(pil_image)
                            
                            # Get image description if enabled
                            description = ""
                            if use_image_description:
                                description = self._get_image_description(element)
                            
                            # Store both OCR and description
                            image_content_data.append((
                                picture_counter,
                                description,
                                ocr_text
                            ))
                    except Exception as e:
                        self.logger.error(f"Error processing image #{picture_counter}: {str(e)}")
            
            # Generate markdown with image references
            temp_md_path = f"{temp_file_path}_temp.md"
            document.save_as_markdown(temp_md_path, image_mode=ImageRefMode.REFERENCED)
            
            # Read the markdown content
            with open(temp_md_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            
            # Find all image references
            image_refs = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_content))
            
            if len(image_refs) == len(image_content_data):
                # Replace image references with their content
                modified_content = markdown_content
                
                for i, (match, (_, description, ocr_text)) in enumerate(zip(image_refs, image_content_data)):
                    # Extract image reference
                    img_ref = match.group(0)
                    
                    # Create replacement content
                    replacement = ""
                    
                    # Add description if available
                    if description:
                        replacement += f"\n\n**Image Description:** {description}\n\n"
                    
                    # Add OCR text if available
                    if ocr_text:
                        replacement += f"\n\n{ocr_text}\n\n"
                    
                    # If no content was extracted, just use a newline
                    if not replacement:
                        replacement = "\n"
                    
                    # Replace in markdown
                    modified_content = modified_content.replace(img_ref, replacement, 1)
                    self.logger.info(f"Replaced image #{i+1} with content")
                
                # self.logger.info(f"Enhanced markdown with content for {len(image_content_data)} images")
                return modified_content
            else:
                self.logger.warning(f"Image references ({len(image_refs)}) and content data ({len(image_content_data)}) don't match in number")
                return markdown_content
        except Exception as e:
            self.logger.error(f"Enhanced extraction error: {str(e)}")
            # Fall back to regular docling extraction
            self.logger.info("Falling back to regular docling extraction")
            return await self.docling_extract(temp_file_path)
    
    def _get_image_description(self, element):
        """Get the SmolVLM description for an image if available"""
        try:
            # Check if the element has annotations
            if hasattr(element, 'annotations') and element.annotations:
                for annotation in element.annotations:
                    # Get description text from annotation
                    if hasattr(annotation, 'text'):
                        description = annotation.text
                        self.logger.info(f"Found image description: {description}")
                        return description
        except Exception as e:
            self.logger.error(f"Error getting image description: {str(e)}")
        
        return ""

    def _perform_ocr_in_memory(self, pil_image):
        """Perform OCR directly on a PIL image in memory (from EnhancedImageProcessor)"""
        try:
            # Convert to numpy array
            img_array = np.array(pil_image)
            
            # Try different RapidOCR return formats
            try:
                result, _, _ = self.ocr_engine(img_array)
            except ValueError:
                try:
                    result, _ = self.ocr_engine(img_array)
                except ValueError:
                    result = self.ocr_engine(img_array)
            
            # Process OCR results
            ocr_text = ""
            if result:
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, (list, tuple)) and len(item) > 1:
                            ocr_text += item[1] + "\n"
                        else:
                            ocr_text += str(item) + "\n"
                else:
                    ocr_text = str(result)
                
                return ocr_text.strip()
            else:
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {str(e)}")
            return ""
    
    def validate_file_extension(self, file: UploadFile):
        extension = file.filename.split(".")[-1].lower()
        if extension in SUPPORTED_FORMATS:
            if extension in FormatToExtensions["pdf"]:
                return "pdf"
            elif extension in FormatToExtensions["image"]:
                return "image"
            else:
                return "other"
        return "unsupported"