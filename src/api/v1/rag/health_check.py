from src.core.utils.config_loader import ConfigReaderInstance
from fastapi import status
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse

from src.app import logger_instance
from src.core.utils.config import settings
# from src.core.utils.config_loader import ConfigReaderInstance


router = APIRouter()
logger = logger_instance.get_logger(__name__)
api_config = ConfigReaderInstance.yaml.read_config_from_file(settings.API_CONFIG_FILENAME)


@router.get('/ping', responses={200: {
            'description': 'Healthcheck Service',
            'content': {
                'application/json': {
                    'example': {'REVISION': '1.0.0'}
                }
            }
        }})

async def health_check() -> JSONResponse:
    logger.info('event=health-check-success message="Successful health check. "')
    content = {'REVISION': api_config.get('API_VERSION')}
    return JSONResponse(content=content, status_code=status.HTTP_200_OK)
