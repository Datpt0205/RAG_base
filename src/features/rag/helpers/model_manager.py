import aiohttp
import asyncio
from typing import List, Dict, Any
from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.utils.config import settings
from src.features.rag.helpers.singleton_helper import SingletonMeta

class ModelManager(LoggerMixin, metaclass=SingletonMeta):
    """
    Manage testing and loading models from Ollama
    """
    DEFAULT_MODELS = [
        # "deepseek-r1:14b",
        # "qwen2.5:14b",
        # "qwen3:14b",
        # "qwen3:1.7b",
        # "qwen3:4b",
        # "llama3.1:8b",
        # "aya:latest"
    ]
    
    def __init__(self):
        super().__init__()
        self.loaded_models = set()
    
    async def ensure_model_pulled(self, model_name: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                # Check if the model is already in Ollama
                async with session.get(f"{settings.OLLAMA_ENDPOINT}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get('name') for model in data.get('models', [])]
                        
                        if model_name in models:
                            self.logger.info(f"Model {model_name} is already available in Ollama")
                            return True
                        
                        # If the model does not exist, perform a pull
                        self.logger.info(f"Model {model_name} does not exist yet, proceed to pull...")
                        async with session.post(
                            f"{settings.OLLAMA_ENDPOINT}/api/pull", 
                            json={"name": model_name}
                        ) as pull_response:
                            if pull_response.status == 200:
                                self.logger.info(f"Pulled model {model_name} successfully")
                                return True
                            else:
                                pull_error = await pull_response.text()
                                self.logger.error(f"Error when pulling model: {pull_error}")
                                return False
                    else:
                        error = await response.text()
                        self.logger.error(f"Unable to get model list from Ollama: {error}")
                        return False
        except Exception as e:
            self.logger.error(f"Error when checking/pulling model {model_name}: {str(e)}")
            return False
    
    async def load_model(self, model_name: str) -> bool:
        """Load the model into memory ready for use"""
        try:
            if model_name in self.loaded_models:
                self.logger.info(f"Model {model_name} has been loaded before")
                return True
            
            import torch
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            self.logger.info(f"Loading model {model_name} into memory using {device_type}...")
            
            is_pulled = await self.ensure_model_pulled(model_name)
            if not is_pulled:
                self.logger.error(f"Cannot pull model {model_name}, cannot load")
                return False
            
            # Load model
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.OLLAMA_ENDPOINT}/api/generate", 
                    json={
                        "model": model_name,
                        "prompt": "hello",
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Model {model_name} has been loaded successfully")
                        self.loaded_models.add(model_name)
                        return True
                    else:
                        error = await response.text()
                        self.logger.error(f"Error loading model {model_name}: {error}")
                        return False
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    async def load_default_models(self) -> Dict[str, bool]:
        results = {}
        for model in self.DEFAULT_MODELS:
            results[model] = await self.load_model(model)
        return results

# Create singleton instance
model_manager = ModelManager()