import time
import datetime 
from typing import List, Tuple, Optional
from operator import itemgetter

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from typing import Optional, Dict, List, Any, Tuple
from collections.abc import AsyncGenerator

from src.core.schemas.response import BasicResponse, BasicResponseDelete
from src.core.database.models.schemas import ChatSessions
from src.core.database.repository.file_repository import FileProcessingRepository
from src.features.rag.handlers.vector_store_handler import VectorStoreQdrant
from src.features.rag.handlers.multi_collection_retriever import multi_collection_retriever
from src.features.rag.handlers.retrieval_handler import SearchRetrieval
from src.features.rag.helpers.llm_helper import LLMGenerator, LLMGeneratorProvider
from src.features.rag.helpers.prompt_template_helper import ContextualizeQuestionHistoryTemplate, QuestionAnswerTemplate
from src.features.rag.helpers.chat_management_helper import ChatService
from src.features.rag.helpers.qdrant_connection_helper import QdrantConnection
from src.features.rag.agents.react_agent import PlanningModule
from src.features.menu_extraction.prompts.prompt_manager import prompt_manager
from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.providers.provider_factory import ProviderType
from src.core.utils.config import settings

from src.features.rag.ai_agents.langgraph_chat_agent import LangGraphChatAgent, AgentState
from langchain_core.messages import HumanMessage

# Initialize the chat service
chat_service = ChatService()

class ChatHandler(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()

        self.search_retrieval = SearchRetrieval()
        self.llm_generator = LLMGenerator()
        self.llm_generator_provider = LLMGeneratorProvider()
    

    # Create a new chat session
    def create_session_id(self, user_id: str, organization_id: Optional[str] = None) -> BasicResponse:
        try:
            session_id = chat_service.create_chat_session(
                user_id=user_id,
                organization_id=organization_id
            )

            return BasicResponse(
                status="Success",
                message="Session created successfully",
                data=session_id
            )
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to create session: {str(e)}",
                data=None
            )
        

    # Delete a chat session
    async def delete_session_completely(
        self, 
        session_id: str, 
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        delete_documents: bool = False,
        delete_collections: bool = False
    ) -> BasicResponseDelete:
        """
        Delete a chat session completely with options to delete related documents and collections
        
        Args:
            session_id: The ID of the chat session to delete
            user_id: The ID of the user (for permissions)
            organization_id: The ID of the organization (for permissions)
            delete_documents: Whether to delete related documents
            delete_collections: Whether to delete related collections
            
        Returns:
            BasicResponseDelete: Response indicating success or failure
        """
        try:                       
            # Delete session and get info about related resources
            result = chat_service.delete_chat_session_completely(
                session_id=session_id,
                delete_documents=delete_documents,
                delete_collections=delete_collections,
                organization_id=organization_id
            )
            
            if result["status"] != "success":
                return BasicResponseDelete(
                    Status="Failed",
                    Message=result["message"],
                    Data=None
                )
            
            deleted_items = result["deleted_items"]
            collections_docs = result.get("collections_docs", {})
            
            # Handle document deletion if needed
            if delete_documents and collections_docs:
                file_management = FileProcessingRepository()
                
                for collection_name, doc_ids in collections_docs.items():
                    # Delete documents from PostgreSQL
                    for doc_id in doc_ids:
                        file_management.delete_file_record(doc_id, organization_id)
                        deleted_items["documents"].append(doc_id)
                    
                    # Delete documents from vector store
                    qdrant_client = QdrantConnection()
                    
                    await qdrant_client.delete_document_by_batch_ids(
                        document_ids=doc_ids,
                        collection_name=collection_name,
                        organization_id=organization_id
                    )
            
            # Handle collection deletion if needed
            if delete_collections and collections_docs:
                vector_store = VectorStoreQdrant()
                
                for collection_name in collections_docs.keys():
                    result = vector_store.delete_qdrant_collection(
                        collection_name=collection_name,
                        user={"id": user_id},
                        organization_id=organization_id,
                        is_personal=(organization_id is None)
                    )
                    if result.status == "Success":
                        deleted_items["collections"].append(collection_name)
            
            return BasicResponseDelete(
                Status="Success",
                Message="Chat session deleted successfully",
                Data=deleted_items
            )
                
        except Exception as e:
            self.logger.error(f"Failed to delete chat session completely: {str(e)}")
            return BasicResponseDelete(
                Status="Failed",
                Message=f"Failed to delete chat session: {str(e)}",
                Data=None
            )


    # Get chat flow for context retrieval and response generation
    async def _get_chat_flow(self, model_name: str, collection_name: str, user_id: str = None, organization_id: str = None, use_multi_collection: bool = False) -> Tuple[Runnable, Runnable]:
        """
        Create the chat flow for retrieving context and generating responses
        
        Args:
            model_name: The name of the LLM model to use
            collection_name: The name of the vector collection to query
            user_id: User ID for multi-collection access (optional)
            organization_id: Organization ID for multi-collection access (optional)
            use_multi_collection: Whether to use both personal and organizational collections
            
        Returns:
            Tuple[Runnable, Runnable]: The conversation chain and rewrite chain
        """
        # Get the language model
        llm = await self.llm_generator.get_llm(model=model_name)
        
        # Chain for rewriting the question based on conversation history
        rewrite_prompt = ContextualizeQuestionHistoryTemplate
        rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config(run_name='rewrite_chain')

        # Define the retrieval function
        async def retriever_function(query):
            if use_multi_collection and user_id:
                # Use multi-collection retriever if required
                return await multi_collection_retriever.retrieve_from_collections(
                    query=query, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                # Use a regular retriever
                return await self.search_retrieval.qdrant_retrieval(
                    query=query, 
                    collection_name=collection_name
                )
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Main conversation chain that combines the rewritten query, context, and generates a response
        chain = (
            {
                "context": itemgetter("rewrite_input") | RunnableLambda(retriever_function).with_config(run_name='stage_retrieval') | format_docs,
                "input": itemgetter("input")
            }
            | QuestionAnswerTemplate
            | llm
            | StrOutputParser()
        ).with_config(run_name='conversational_rag')
        
        return chain, rewrite_chain

    def _get_api_key(self, provider_type: str) -> Optional[str]:
        """
        Get API key for a provider from environment variables.
        
        Args:
            provider_type: Provider type (ollama, openai, gemini)
            
        Returns:
            Optional[str]: API key for the provider
        """
        if provider_type == ProviderType.OPENAI:
            return settings.OPENAI_API_KEY
        elif provider_type == ProviderType.GEMINI:
            return settings.GEMINI_API_KEY
        elif provider_type == ProviderType.OLLAMA:
            return settings.OLLAMA_ENDPOINT
        
        return None


    async def classify_query(self, query: str, model_name: str = "llama3.1:8b") -> str:
        """
        Classify a query to determine if it needs retrieval or can be answered directly.
        
        Args:
            query: User query
            model_name: LLM model name
            
        Returns:
            str: Either "DIRECT" or "RETRIEVAL"
        """
        try:
            # Get LLM
            llm = await self.llm_generator.get_llm(model=model_name)
            
            # Format classifier message
            classifier_template = prompt_manager.get_template("query_classifier")
            classifier_message = classifier_template.format(query=query)
            
            # Use system + user message format
            messages = [
                {"role": "system", "content": "You are a query classifier assistant."},
                {"role": "user", "content": classifier_message}
            ]
            
            # Get classification
            response = await llm.ainvoke(messages)
            classification = response.content if hasattr(response, 'content') else str(response)
            
            # Normalize and extract classification result
            classification = classification.strip().upper()
            if "DIRECT" in classification:
                return "DIRECT"
            else:
                return "RETRIEVAL"
                
        except Exception as e:
            self.logger.error(f"Error classifying query: {str(e)}")
            # Default to retrieval if classification fails
            return "RETRIEVAL"
    

    async def handle_smart_chat(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = False,
        enable_reasoning: bool = False
    ) -> BasicResponse:
        """
        Smart chat router that decides whether to use retrieval or answer directly.
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            enable_reasoning: Whether to enable reasoning mode for complex questions
            
        Returns:
            BasicResponse: Response with generated text
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)
            
            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Classify query - decide if we need retrieval or direct answer
            classify_query_model = "llama3.1:8b"
            query_type = await self.classify_query(question_input, classify_query_model)
            self.logger.info(f"Query classification for '{question_input}': {query_type}")
            
            content = ""
            retrieval_performed = False
            context_docs = []
            
            if query_type == "DIRECT":
                # Direct response without retrieval
                self.logger.info(f"Using direct response for query: '{question_input}'")
                
                if enable_reasoning:
                    # Use reasoning for direct responses if enabled
                    response = await self.llm_generator_provider.generate_react_cot_response(
                        model_name=model_name,
                        query=question_input,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        enable_thinking=enable_thinking
                    )
                else:
                    # Use standard chat template for direct responses
                    response = await self.llm_generator_provider.generate_with_template(
                        model_name=model_name,
                        system_template="chat_system",
                        user_content=question_input,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        enable_thinking=enable_thinking
                    )
                
                content = response["content"]
                
            else:
                # Retrieval-based response
                self.logger.info(f"Using retrieval for query: '{question_input}'")
                
                # Retrieve context if needed
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=question_input, 
                    collection_name=collection_name,
                    top_k=5
                )
                
                # Format context from retrieved documents
                context = ""
                if context_docs:
                    context = "\n\n".join(doc.page_content for doc in context_docs)
                    retrieval_performed = True
                    self.logger.info(f"Retrieved {len(context_docs)} documents for context")
                
                if enable_reasoning:
                    # Use reasoning with RAG if enabled
                    response = await self.llm_generator_provider.generate_react_cot_response(
                        model_name=model_name,
                        query=question_input,
                        context=context,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        enable_thinking=enable_thinking
                    )
                else:
                    # Use RAG approach with context
                    response = await self.llm_generator_provider.generate_rag_response(
                        model_name=model_name,
                        query=question_input,
                        context=context,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        enable_thinking=enable_thinking
                    )
                
                content = response["content"]
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=content,
                response_time=response_time
            )
            
            # Save document references if retrieval was performed
            if retrieval_performed and context_docs:
                await self._save_document_references(message_id, context_docs)
            
            return BasicResponse(
                status='Success',
                message=f"Chat request processed successfully (query_type={query_type})",
                data={
                    "content": content,
                    "query_type": query_type,
                    "retrieval_performed": retrieval_performed
                }
            )
                
        except Exception as e:
            self.logger.error(f"Failed to handle smart chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle smart chat request: {str(e)}",
                data=None
            )
    
    async def handle_smart_chat_stream(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = False,
        enable_reasoning: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Smart chat router with streaming that decides whether to use retrieval or answer directly.
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            enable_reasoning: Whether to enable reasoning mode for complex questions
            
        Yields:
            str: Response chunks
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)

            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Classify query - decide if we need retrieval or direct answer
            classify_query_model = "llama3.1:8b"
            query_type = await self.classify_query(question_input, classify_query_model)
            self.logger.info(f"Query classification for '{question_input}': {query_type}")
            
            # Optional - tell user what mode we're using
            yield f"[{query_type} MODE]\n\n"
            
            retrieval_performed = False
            context_docs = []
            full_response = []
            
            if query_type == "DIRECT":
                # Direct response without retrieval
                self.logger.info(f"Using direct response for query: '{question_input}'")
                
                if enable_reasoning:
                    # Stream with reasoning for direct responses
                    async for chunk in self.llm_generator_provider.stream_react_cot_response(
                        model_name=model_name,
                        query=question_input,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        clean_thinking=not enable_reasoning,
                        enable_thinking=enable_thinking
                    ):
                        full_response.append(chunk)
                        yield chunk
                else:
                    # Stream with standard template
                    async for chunk in self.llm_generator_provider.stream_with_template(
                        model_name=model_name,
                        system_template="chat_system",
                        user_content=question_input,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        enable_thinking=enable_thinking
                    ):
                        full_response.append(chunk)
                        yield chunk
                
            else:
                # Retrieval-based response
                self.logger.info(f"Using retrieval for query: '{question_input}'")
                
                # Yield a notification that retrieval is happening
                yield "Searching knowledge base...\n\n"
                
                # Retrieve context
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=question_input, 
                    collection_name=collection_name,
                    top_k=5
                )
                
                # Format context from retrieved documents
                context = ""
                if context_docs:
                    context = "\n\n".join(doc.page_content for doc in context_docs)
                    retrieval_performed = True
                    self.logger.info(f"Retrieved {len(context_docs)} documents for context")
                
                if enable_reasoning:
                    # Stream with reasoning and RAG
                    async for chunk in self.llm_generator_provider.stream_react_cot_response(
                        model_name=model_name,
                        query=question_input,
                        context=context,
                        history_messages=history_messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        clean_thinking=not enable_reasoning, 
                        enable_thinking=enable_thinking
                    ):
                        full_response.append(chunk)
                        yield chunk
                else:
                    # Format RAG messages # Need to add thinking_instruction?
                    messages = prompt_manager.format_rag_messages(
                        query=question_input,
                        context=context,
                        history_messages=history_messages 
                    )
                    
                    # Stream responses
                    async for chunk in self.llm_generator_provider.stream_response(
                        model_name=model_name,
                        messages=messages,
                        provider_type=provider_type,
                        api_key=api_key,
                        clean_thinking=True
                    ):
                        full_response.append(chunk)
                        yield chunk
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content="".join(full_response),
                response_time=response_time
            )
            
            # Save document references if retrieval was performed
            if retrieval_performed and context_docs:
                await self._save_document_references(message_id, context_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to handle streaming smart chat request: {str(e)}")
            yield f"An error occurred: {str(e)}"



    # API /chat
    async def handle_request_chat(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False
    ) -> BasicResponse:
        """
        Handle a chat request: retrieve context, generate a response
        
        Args:
            session_id: The chat session ID
            question_input: The user's question
            model_name: The LLM model to use
            collection_name: The vector collection to query
            user_id: The user ID for multi-collection access
            organization_id: The organization ID for multi-collection access
            use_multi_collection: Whether to use both personal and organizational collections
            
        Returns:
            BasicResponse: The response to the chat request
        """
        try:
            # Get the chains needed for the chat flow
            conversational_rag_chain, rewrite_chain = await self._get_chat_flow(
                model_name=model_name, 
                collection_name=collection_name,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=use_multi_collection
            )

            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # # # COUNT TOKENS
            # from src.features.rag.helpers.llm_helper import count_tokens
            # input_tokens_without_context = count_tokens(question_input)
            # self.logger.info(f"Input tokens without context: {input_tokens_without_context}")
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history and rewrite the question for better context
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            rewrite_input = await rewrite_chain.ainvoke(
                input={"input": question_input, "chat_history": chat_history}
            )
            
            # Retrieve context documents
            context_docs = []
            if use_multi_collection and user_id:
                context_docs = await multi_collection_retriever.retrieve_from_collections(
                    query=rewrite_input, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=rewrite_input, 
                    collection_name=collection_name
                )
            
            # Format context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
            # self.logger.info(f"Length context for LLM model {len(context)}")
            # self.logger.info(f"Length context for LLM model {(context)}")

            # # COUNT TOKENS
            # prompt_with_context = f"Context: {context}\n\nQuestion: {question_input}"
            # input_tokens_with_context = count_tokens(prompt_with_context)
            # self.logger.info(f"Input tokens with context: {input_tokens_with_context}")

            # Generate the response with the context
            # Format messages using template
            messages = QuestionAnswerTemplate.format_messages(
                context=context,
                input=question_input
            )
            
            # Get LLM
            llm = await self.llm_generator.get_llm(model=model_name)
            
            # Generate response
            llm_response = await llm.ainvoke(messages)
            raw_resp = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            resp = self.llm_generator.clean_thinking(raw_resp)
            # resp = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # # COUNT TOKENS
            # output_tokens = count_tokens(resp)
            # self.logger.info(f"Output tokens: {output_tokens}")

            # Calculate the response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=resp,
                response_time=response_time
            )
            
            # # COUNT TOKENS
            # self.logger.info(f"Session {session_id} - Total tokens: input without context={input_tokens_without_context}, input with context={input_tokens_with_context}, output={output_tokens}, total={input_tokens_with_context + output_tokens}")
            
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
            # self.logger.info(f"Successfully handled chat request in session {session_id}")
            
            return BasicResponse(
                status='Success',
                message="Chat request processed successfully",
                data=resp
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle chat request: {str(e)}",
                data=None
            )


    # API /chat/completions   
    async def handle_streaming_chat(
            self,
            session_id: str,
            question_input: str,
            model_name: str,
            collection_name: str,
            user_id: str = None,
            organization_id: str = None,
            use_multi_collection: bool = False
        ) -> AsyncGenerator[str, None]:
        """
        Process chat request and return responses in streaming format
        
        Args:
            session_id: Chat session ID
            question_input: User's question
            model_name: LLM model to use
            collection_name: Vector collection name
            user_id: User ID for access control
            organization_id: Organization ID for access control
            use_multi_collection: Whether to use multiple collections
            
        Yields:
            Response chunks as they're generated
        """
        try:
            # Save user question to database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create placeholder for assistant response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Get LLM for context question rewriting
            llm = await self.llm_generator.get_llm(model=model_name)
            rewrite_chain = ContextualizeQuestionHistoryTemplate | llm | StrOutputParser()
            
            # Rewrite question for better context
            rewrite_input = await rewrite_chain.ainvoke(
                {"input": question_input, "chat_history": chat_history}
            )
            
            # Retrieve relevant context from vector database
            context_docs = []
            if use_multi_collection and user_id:
                context_docs = await multi_collection_retriever.retrieve_from_collections(
                    query=rewrite_input, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=rewrite_input, 
                    collection_name=collection_name
                )
            
            # Format context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
            
            # Format messages using template
            messages = QuestionAnswerTemplate.format_messages(
                context=context,
                input=question_input
            )
            
            # Get streaming-enabled LLM
            streaming_llm = await self.llm_generator.get_llm(model=model_name)
            
            # Store complete response for database update
            full_response = []
            
            # # Stream the response chunks
            # async for chunk in streaming_llm.astream(messages):
            #     if hasattr(chunk, 'content') and chunk.content:
            #         # Clean thinking sections
            #         cleaned_chunk = self.llm_generator.clean_thinking(chunk.content)
            #         if cleaned_chunk:
            #             full_response.append(cleaned_chunk)
            #             yield cleaned_chunk

            # Stream the response chunks directly
            async for chunk in streaming_llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    chunk_content = chunk.content
                    full_response.append(chunk_content)
                    # Yield directly without any cleaning
                    yield chunk_content
                    
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update complete response in database
            complete_response = "".join(full_response)
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=complete_response,
                response_time=response_time
            )
            
            # Save document references safely
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
        except Exception as e:
            self.logger.error(f"Streaming chat error: {str(e)}")
            yield f"An error occurred: {str(e)}"
    
    async def _save_document_references(self, message_id: str, context_docs: List) -> None:
        """
        Safely save document references, checking for duplicates and existence in database
        
        Args:
            message_id: Message ID to associate references with
            context_docs: Context documents from vector search
        """
        try:
            # Track document IDs that have been processed for this message
            processed_doc_ids = set()
            
            # Process each document, ensuring we only process each once
            for doc in context_docs:
                if 'document_id' in doc.metadata:
                    document_id = doc.metadata['document_id']
                    
                    # Skip if already processed for this message
                    if document_id in processed_doc_ids:
                        continue
                    
                    # Mark as processed
                    processed_doc_ids.add(document_id)
                    
                    # Get page number from metadata
                    page = doc.metadata.get('index', 0)
                    
                    # Try to save, ChatService.save_reference_docs will handle the checks
                    result = chat_service.save_reference_docs(
                        message_id=message_id,
                        document_id=document_id,
                        page=page
                    )
                    
                    # Use result to avoid logging for each failure
                    if result is None:
                        # Reference wasn't saved (already exists or document not found)
                        pass
                    
        except Exception as e:
            self.logger.error(f"Error saving document references: {str(e)}")


    # API /chat/provider
    async def handle_request_chat_with_provider(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = True
    ) -> BasicResponse:
        """
        Handle a chat request with specific provider and prompt management.
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            
        Returns:
            BasicResponse: Response with generated text
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)

            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Retrieve context if needed
            context_docs = await self.search_retrieval.qdrant_retrieval(
                query=question_input, 
                collection_name=collection_name,
                top_k=5
            )
            
            # Format context from retrieved documents
            context = ""
            if context_docs:
                context = "\n\n".join(doc.page_content for doc in context_docs)
                self.logger.info(f"Retrieved {len(context_docs)} documents for context")
            
            # Generate response using appropriate method based on context availability
            if context:
                # Use RAG approach with context
                response = await self.llm_generator_provider.generate_rag_response(
                    model_name=model_name,
                    query=question_input,
                    context=context,
                    history_messages=history_messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    enable_thinking=enable_thinking
                )
            else:
                # Use standard chat without context
                response = await self.llm_generator_provider.generate_with_template(
                    model_name=model_name,
                    system_template="chat_system",
                    user_content=question_input,
                    history_messages=history_messages,
                    provider_type=provider_type,
                    api_key=api_key
                )
            
            # Extract content from response
            content = response["content"]
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=content,
                response_time=response_time
            )
            
            # Save document references if available
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
            return BasicResponse(
                status='Success',
                message="Chat request processed successfully",
                data=content
            )
                
        except Exception as e:
            self.logger.error(f"Failed to handle chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle chat request: {str(e)}",
                data=None
            )

    # API /chat/provider/stream
    async def handle_streaming_chat_with_provider(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Handle a streaming chat request with specific provider and prompt management.
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            
        Yields:
            str: Response chunks
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)
            
            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Retrieve context if needed
            context_docs = await self.search_retrieval.qdrant_retrieval(
                query=question_input, 
                collection_name=collection_name,
                top_k=5
            )
            
            # Format context from retrieved documents
            context = ""
            if context_docs:
                context = "\n\n".join(doc.page_content for doc in context_docs)
                self.logger.info(f"Retrieved {len(context_docs)} documents for context")
                
                # For streaming, yield a small notification that context was found
                yield "Searching knowledge base...\n\n"
            
            # Store full response to save in database
            full_response = []
            
            # Stream response using appropriate method based on context availability
            if context:
                # Use RAG approach with context
                messages = prompt_manager.format_rag_messages(
                    query=question_input,
                    context=context,
                    history_messages=history_messages,
                    enable_thinking=enable_thinking
                )
            else:
                # Use standard chat without context
                messages = prompt_manager.format_messages(
                    system_template="chat_system",
                    user_content=question_input,
                    history_messages=history_messages
                )
            
            # Stream responses
            async for chunk in self.llm_generator_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=True
            ):
                full_response.append(chunk)
                yield chunk
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content="".join(full_response),
                response_time=response_time
            )
            
            # Save document references if available
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to handle streaming chat request: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def _convert_history_to_messages(self, chat_history: str) -> List[Dict[str, str]]:
        """
        Convert chat history to messages format.
        
        Args:
            chat_history: Chat history in string format
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        messages = []
        
        if not chat_history:
            return messages
        
        lines = chat_history.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            if " - user: " in line:
                content = line.split(" - user: ", 1)[1]
                messages.append({"role": "user", "content": content})
            elif " - assistant: " in line:
                content = line.split(" - assistant: ", 1)[1]
                messages.append({"role": "assistant", "content": content})
        
        return messages
    

    # API /chat/provider/reasoning
    async def handle_chat_provider_reasoning(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = False
    ) -> BasicResponse:
        """stream_response
        Handle a chat request using ReAct+CoT (Reasoning and Acting with Chain of Thought).
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            enable_thinking: Whether to preserve <thinking> tags in the final output
            
        Returns:
            BasicResponse: Response with generated text
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)
            
            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Retrieve context if needed
            context_docs = await self.search_retrieval.qdrant_retrieval(
                query=question_input, 
                collection_name=collection_name,
                top_k=5
            )
            
            # Format context from retrieved documents
            context = ""
            if context_docs:
                context = "\n\n".join(doc.page_content for doc in context_docs)
                self.logger.info(f"Retrieved {len(context_docs)} documents for context")
            
            # Generate response using ReAct+CoT
            response = await self.llm_generator_provider.generate_react_cot_response(
                model_name=model_name,
                query=question_input,
                context=context,
                history_messages=history_messages,
                provider_type=provider_type,
                api_key=api_key,
                enable_thinking=enable_thinking
            )
            
            # Extract content from response
            content = response["content"]
            
            # Clean thinking tags if not preserved
            if not enable_thinking:
                content = self.llm_generator_provider.clean_thinking(content)
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=content,
                response_time=response_time
            )
            
            # Save document references if available
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
            return BasicResponse(
                status='Success',
                message="Chat request processed successfully with ReAct+CoT",
                data=content
            )
                
        except Exception as e:
            self.logger.error(f"Failed to handle ReAct+CoT chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle ReAct+CoT chat request: {str(e)}",
                data=None
            )


    # API /chat/provider/reasoning/stream
    async def handle_chat_provider_reasoning_stream(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        clean_thinking: bool = True,
        enable_thinking: bool = True 
    ) -> AsyncGenerator[str, None]:
        """
        Handle a streaming chat request using ReAct+CoT.
        
        Args:
            session_id: Chat session ID
            question_input: User question
            model_name: LLM model name
            collection_name: Collection name in vector database
            provider_type: Provider type (ollama, openai, gemini)
            api_key: API key for paid providers
            user_id: User ID
            organization_id: Organization ID
            use_multi_collection: Whether to use multiple collections
            clean_thinking: Whether to remove <thinking> tags from output
            
        Yields:
            str: Response chunks
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)
            
            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Convert chat history to messages format
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Retrieve context if needed
            context_docs = await self.search_retrieval.qdrant_retrieval(
                query=question_input, 
                collection_name=collection_name,
                top_k=5
            )
            
            # Format context from retrieved documents
            context = ""
            if context_docs:
                context = "\n\n".join(doc.page_content for doc in context_docs)
                self.logger.info(f"Retrieved {len(context_docs)} documents for context")
                
                # For streaming, yield a small notification that context was found
                yield "Searching knowledge base...\n\n"
            
            # Store full response to save in database
            full_response = []
            
            # Stream response using ReAct+CoT
            async for chunk in self.llm_generator_provider.stream_react_cot_response(
                model_name=model_name,
                query=question_input,
                context=context,
                history_messages=history_messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=clean_thinking,
                enable_thinking=enable_thinking
            ):
                full_response.append(chunk)
                print(f"Original chunk: '{chunk}'")
                yield chunk
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Process full response to clean thinking if needed
            complete_response = "".join(full_response)
            if clean_thinking:
                complete_response = self.llm_generator_provider.clean_thinking(complete_response)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=complete_response,
                response_time=response_time
            )
            
            # Save document references if available
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to handle streaming ReAct+CoT chat request: {str(e)}")
            yield f"An error occurred: {str(e)}"

    
    async def handle_chat_agent_provider_reasoning(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        enable_thinking: bool = False
    ) -> BasicResponse:
        """
        Handle chat với LangGraph Agent - tool-based retrieval
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)
            
            # Save user question
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create placeholder response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )
            
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Create agent
            agent = LangGraphChatAgent(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key, 
                search_retrieval=self.search_retrieval,
                multi_collection_retriever=multi_collection_retriever,
                enable_thinking=enable_thinking
            )
            
            # Prepare initial state
            initial_messages = []
            
            # Add history messages
            for msg in history_messages:
                if msg["role"] == "user":
                    initial_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    initial_messages.append(AIMessage(content=msg["content"]))
            
            # Add current question
            initial_messages.append(HumanMessage(content=question_input))
            
            # Create state
            state = {
                "messages": initial_messages,
                "session_id": session_id,
                "collection_name": collection_name,
                "use_multi_collection": use_multi_collection,
                "user_id": user_id,
                "organization_id": organization_id,
                "retrieved_docs": None
            }
            
            # Invoke agent
            result = await agent.invoke(state)
            
            # Get response and docs
            content = result["response"]
            retrieved_docs = result.get("retrieved_docs", [])
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update response in database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=content,
                response_time=response_time
            )
            
            # Save document references if any
            if retrieved_docs:
                await self._save_document_references(message_id, retrieved_docs)
                self.logger.info(f"Agent used retrieval and found {len(retrieved_docs)} documents")
            else:
                self.logger.info("Agent answered without retrieval")
            
            return BasicResponse(
                status='Success',
                message="Chat request processed successfully with LangGraph Agent",
                data=content
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle LangGraph chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle chat request: {str(e)}",
                data=None
            )


class ChatMessageHistory(LoggerMixin):
    """
    Utility class for working with chat message history
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def messages_from_items(items: list) -> List[BaseMessage]:
        """
        Convert raw message items to BaseMessage objects
        
        Args:
            items: List of (content, type) tuples
            
        Returns:
            List[BaseMessage]: List of message objects
        """
        def _message_from_item(message: tuple) -> BaseMessage:
            _type = message[1]
            if _type == "human" or _type == "user":
                return HumanMessage(content=message[0])
            elif _type == "ai" or _type == "assistant":
                return AIMessage(content=message[0])
            elif _type == "system":
                return SystemMessage(content=message[0])
            else:
                raise ValueError(f"Got unexpected message type: {_type}")

        messages = [_message_from_item(msg) for msg in items]
        return messages

    @staticmethod
    def concat_message(messages: List[BaseMessage]) -> str:
        """
        Concatenate messages into a single string
        
        Args:
            messages: List of BaseMessage objects
            
        Returns:
            str: Concatenated message history
        """
        concat_chat = ""
        for mes in messages:
            if isinstance(mes, HumanMessage):
                concat_chat += " - user: " + mes.content + "\n"
            else:
                concat_chat += " - assistant: " + mes.content + "\n"
        return concat_chat
    
    @staticmethod
    def string_message_chat_history(session_id: str) -> str:
        """
        Get the chat history as a string
        
        Args:
            session_id: The ID of the chat session
            
        Returns:
            str: The chat history as a string
        """
        items = chat_service.get_chat_history(session_id=session_id, limit=2)
        messages = ChatMessageHistory.messages_from_items(items)
        
        # Reverse the order and skip the current message being processed
        history_str = ChatMessageHistory.concat_message(messages[::-1][:-2])
        return history_str

    def get_list_message_history(
        self, 
        session_id: str, 
        limit: int = 10, 
        user_id: Optional[str] = None, 
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Get the list of messages in the chat history
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response with message history as data
        """
        try:
            # Check access if user_id is provided
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Check if user_id matches session owner
                    if session_info.get("user_id") != user_id:
                        # Check if the user belongs to the organization that owns the session
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Organizational users, allowing access
                            pass
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to view this chat history",
                                data=None
                            )
            
            # Get chat history
            items = chat_service.get_chat_history(session_id=session_id, limit=limit)
            
           # Format the items as "{role} : {content}"
            formatted_items = [f"{item[1]} : {item[0]}" for item in items]
            
            return BasicResponse(
                status="Success",
                message="Retrieved message history successfully",
                data=formatted_items
            )
        
        except Exception as e:
            self.logger.error(f"Failed to get message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to get message history: {str(e)}",
                data=None
            )
    

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information of a chat session
        
        Args:
            session_id: ID chat session
            
        Returns:
            Optional[Dict[str, Any]]: Session information or None if not found
        """
        try:
            from src.core.database.db_connection import db
            
            with db.session_scope() as session:
                chat_session = session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).first()
                
                if not chat_session:
                    return None
                    
                return {
                    "id": str(chat_session.id),
                    "user_id": chat_session.user_id,
                    "organization_id": chat_session.organization_id,
                    "title": chat_session.title,
                    "start_date": chat_session.start_date
                }
        except Exception as e:
            self.logger.error(f"Failed to get session info: {str(e)}")
            return None
                          
    def delete_message_history(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Delete the chat history for a session
        
        Args:
            session_id: The ID of the chat session to delete
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response indicating success or failure
        """
        try:
            # Check delete permission if user_id is provided
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Check if user_id matches session owner
                    if session_info.get("user_id") != user_id:
                        # Check if user has admin rights in the organization
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Need to check admin role here if possible
                            from src.features.rag.handlers.user_role_handler import UserRoleService
                            user_role_service = UserRoleService()

                            is_admin = user_role_service.is_admin(user_id, organization_id)
                            if not is_admin:
                                return BasicResponse(
                                    status="Failed",
                                    message="You don't have permission to delete this chat history",
                                    data=None
                                )
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to delete this chat history",
                                data=None
                            )
            
            if chat_service.is_session_exist(session_id):
                chat_service.delete_chat_history(session_id=session_id)
                return BasicResponse(
                    status="Success",
                    message="Chat history deleted successfully",
                    data=None
                )
            else:
                return BasicResponse(
                    status="Failed",
                    message="Chat session does not exist",
                    data=None
                )
            
        except Exception as e:
            self.logger.error(f"Failed to delete message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to delete message history: {str(e)}",
                data=None
            )