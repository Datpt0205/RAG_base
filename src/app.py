class LoggerInstance(object):
    def __new__(cls):
        from src.core.utils.logger.custom_logging import LogHandler
        return LogHandler()

class IncludeAPIRouter(object):
    def __new__(cls):
        from fastapi.routing import APIRouter
        from src.api.v1.rag.health_check import router as router_health_check
        from src.api.v1.rag.security import router as router_security
        from src.api.v1.rag.vectorstore import router as router_collection_management
        from src.api.v1.rag.documents import router as router_document_management
        from src.api.v1.rag.retriever import router as router_retriever
        from src.api.v1.rag.rerank import router as router_rerank
        from src.api.v1.rag.llm_chat import router as router_chatllm
        from src.api.v1.rag.ai_translation import router as router_translation
        
        # RateMate
        from src.api.v1.menu_extraction.review_generator import router as router_review_generator
        from src.api.v1.menu_extraction.menu_extractor import router as router_menu_extractor

        router = APIRouter(prefix='/api/v1')
        router.include_router(router_health_check, tags=['Health Check'])
        router.include_router(router_security, tags=['Security'])
        router.include_router(router_collection_management, tags=['Collection Management'])
        router.include_router(router_document_management, tags=['Document Management'])
        router.include_router(router_retriever, tags=['Retriever'])
        router.include_router(router_rerank, tags=['Reranking'])
        router.include_router(router_chatllm, tags=['LLM Conversation'])
        router.include_router(router_translation, tags=['AI Translation'])
        
        # RateMate 
        router.include_router(router_review_generator, tags=['RateMate - Review Generator'])
        router.include_router(router_menu_extractor, tags=['RateMate - Menu Extractor'])

        return router

# Instance creation
logger_instance = LoggerInstance()