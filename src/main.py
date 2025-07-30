import os
import logging
import uvicorn
import secrets
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from src.core.utils.config import settings
from src.core.utils.constants import CODEMIND_LLM
from src.app import IncludeAPIRouter, logger_instance
from src.core.utils.config_loader import ConfigReaderInstance
from src.features.rag.handlers.file_partition_handler import DocumentExtraction

logger = logger_instance.get_logger(__name__)

# Read configuration from YAML file
api_config = ConfigReaderInstance.yaml.read_config_from_file(settings.API_CONFIG_FILENAME)
logging_config = ConfigReaderInstance.yaml.read_config_from_file(settings.LOG_CONFIG_FILENAME)

# Generate a security key (used to encrypt the session).
secret_key = secrets.token_urlsafe(32)

# lifespan (app lifecycle management, default is None).
def get_application(lifespan: Any = None):
    _app = FastAPI(lifespan=lifespan,
                   title=api_config.get('API_NAME'),
                   description=api_config.get('API_DESCRIPTION'),
                   version=api_config.get('API_VERSION'),
                   debug=api_config.get('API_DEBUG_MODE')
                   )
    
    _app.include_router(IncludeAPIRouter())

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.add_middleware(SessionMiddleware, secret_key=secret_key)

    return _app

# Manage the lifecycle of asynchronous applications.
# Perform actions when the application starts and shuts down
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    logger.info(CODEMIND_LLM)
    logger.info(f'event=app-startup')

    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
         
            torch.set_num_threads(4)
        else:
            logger.warning("CUDA not available. Using CPU for model inference.")
    except Exception as e:
        logger.error(f"Error checking CUDA: {str(e)}")

    try:
        from src.features.rag.helpers.model_manager import model_manager
        
        logger.info("Start loading default models...")
        results = await model_manager.load_default_models()
        
        for model, success in results.items():
            if success:
                logger.info(f"Model {model} loaded successfully")
            else:
                logger.warning(f"Could not load model {model}")
        
        DocumentExtraction.get_instance()

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

    yield
    # Code to execute when app is shutting down
    logger.info(f'event=app-shutdown message="All connections are closed."')


# Create FastAPI application object
app = get_application(lifespan=app_lifespan)

@app.get('/')
async def docs_redirect():
    return RedirectResponse(url='/docs')

if __name__ == '__main__':
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config['formatters']['access']['fmt'] = logging_config.get('UVICORN_FORMATTER')
    log_config['formatters']['default']['fmt'] = logging_config.get('UVICORN_FORMATTER')
    log_config['formatters']['access']['datefmt'] = logging_config.get('DATE_FORMATTER')
    log_config['formatters']['default']['datefmt'] = logging_config.get('DATE_FORMATTER')
    
    uvicorn.run('src.main:app',
                host=settings.HOST,
                port=settings.PORT,
                log_level=settings.LOG_LEVEL.lower(),
                log_config=log_config,
                workers=settings.UVICORN_WORKERS,
               )
