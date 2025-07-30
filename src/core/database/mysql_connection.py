import pymysql
import pymysql.cursors
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional

from src.core.utils.config import settings
from src.core.utils.config_loader import ConfigReaderInstance
from src.features.rag.helpers.singleton_helper import SingletonMeta
from src.core.utils.logger.custom_logging import LoggerMixin

# Load MySQL configuration
mysql_config = ConfigReaderInstance.yaml.read_config_from_file("database_config.yaml")
MYSQL_CONFIG = mysql_config.get('MYSQL', {})


class MySQLConnection(LoggerMixin, metaclass=SingletonMeta):
    """
    Singleton class for managing MySQL connections to the Frontend database
    """
    def __init__(self):
        super().__init__()
        self.connection_params = {
            'host': MYSQL_CONFIG.get('HOST', 'localhost'),
            'port': int(MYSQL_CONFIG.get('PORT', 3306)),
            'user': MYSQL_CONFIG.get('USER', 'root'),
            'password': MYSQL_CONFIG.get('PASSWORD', ''),
            'db': MYSQL_CONFIG.get('DATABASE', 'frontend_db'),
            'charset': MYSQL_CONFIG.get('CHARSET', 'utf8mb4'),
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.logger.info(f"Initialized MySQL connection to {self.connection_params['host']}:{self.connection_params['port']}")
        
    def get_connection(self) -> pymysql.connections.Connection:
        """
        Get a connection to the MySQL database
        
        Returns:
            pymysql.connections.Connection: Database connection
        """
        try:
            connection = pymysql.connect(**self.connection_params)
            return connection
        except Exception as e:
            self.logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise
    
    @contextmanager
    def connection_scope(self) -> Generator[pymysql.connections.Connection, None, None]:
        """
        Provides a connection context manager for safe usage
        
        Yields:
            Generator[pymysql.connections.Connection, None, None]: Database connection
        """
        connection = self.get_connection()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Error in MySQL connection scope: {str(e)}")
            raise
        finally:
            connection.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute a query and return the result
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Query results
        """
        with self.connection_scope() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return result
    
    def execute_scalar(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query and return a single value
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Any: Query result (single value)
        """
        with self.connection_scope() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else None


# Create a singleton instance
mysql_connection = MySQLConnection()