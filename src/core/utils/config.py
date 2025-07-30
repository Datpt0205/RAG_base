import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path)

APP_HOME = os.environ.get('APP_HOME')

class AppConfig(BaseModel):
    """Application configurations."""

    # Defines the root directory of the application.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Defines the settings directory located in the root directory.
    SETTINGS_DIR: Path = BASE_DIR.joinpath('settings')
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Global configurations."""

    APP_CONFIG: AppConfig = AppConfig()

    # define global variables with the Field class
    ENV_STATE: str = Field('dev', env='ENV_STATE')
    LOG_LEVEL: str = Field('DEBUG', env='LOG_LEVEL')

    HOST: str = Field('0.0.0.0', env='HOST')
    PORT: int = Field('8686', env='PORT')
    
    # Number of workers when running Uvicorn.
    UVICORN_WORKERS: int = Field(1, env='UVICORN_WORKERS')

    API_CONFIG_FILENAME: str = Field('api_config.yaml', env='API_CONFIG_FILENAME')
    LOG_CONFIG_FILENAME: str = Field('logging_config.yaml', env='LOG_CONFIG_FILENAME')
    AUTH_CONFIG_FILENAME: str = Field('auth_config.yaml', env='AUTH_CONFIG_FILENAME')
    DATABASE_CONFIG_FILENAME: str = Field('database_config.yaml', env='DATABASE_CONFIG_FILENAME')

    MODEL_CONFIG_FILENAME: str = Field('model_config.yaml', env='MODEL_CONFIG_FILENAME')

    # Define access token Huggingface
    HUGGINGFACE_ACCESS_TOKEN: str | None = Field(None, env='HUGGINGFACE_ACCESS_TOKEN')

    LLM_MAX_RETRIES: int = Field(3, env='LLM_MAX_RETRIES')

    # Define config for Qdrant
    QDRANT_ENDPOINT: str | None = Field(..., env='QDRANT_ENDPOINT') 
    QDRANT_COLLECTION_NAME: str = Field(..., env='QDRANT_COLLECTION_NAME')

    # MySQL Frontend config
    MYSQL_HOST: str = Field('localhost', env='MYSQL_HOST')
    MYSQL_PORT: int = Field(3306, env='MYSQL_PORT')
    MYSQL_DATABASE: str = Field('frontend_db', env='MYSQL_DATABASE')
    MYSQL_USER: str = Field('frontend_user', env='MYSQL_USER')
    MYSQL_PASSWORD: str = Field('frontend_password', env='MYSQL_PASSWORD')

    # Define config Ollama for hosting model from local
    OLLAMA_ENDPOINT: str = Field(..., env='OLLAMA_ENDPOINT')
    OPENAI_API_KEY: str = Field("", env='OPENAI_API_KEY')
    GEMINI_API_KEY: str = Field("", env='GEMINI_API_KEY')


# Avoid having to re-read the .env file and create the Settings object every time you access it
@lru_cache()
def get_settings():
    return Settings()

# settings will be the object that contains all the configuration of the application.
settings = get_settings()