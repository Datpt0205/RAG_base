import uuid
import secrets
import string
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from src.core.database.repository.api_key_repository import APIKeyRepository
from src.core.database.models.schemas import APIKey
from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.handlers.user_role_handler import UserRoleService

# Testing 
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi import Header

# Header to get API key from request
ORGANIZATION_ID_HEADER = APIKeyHeader(name="X-Organization-Id", auto_error=False)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.api_key_repo = APIKeyRepository()
        self.user_role_service = UserRoleService()

    # Option testing:
    async def author_with_api_key_or_bearer(
        self,
        organization_id: str = Depends(ORGANIZATION_ID_HEADER),
        api_key: str = Depends(API_KEY_HEADER),
        authorization: str = Header(None),
        request: Request = None,
        require_role: str = None
    ) -> Optional[Dict[str, Any]]:
        token_api_key = None
        if authorization and authorization.startswith("Bearer "):
            token_api_key = authorization.replace("Bearer ", "").strip()
            # self.logger.info(f"Found API key in Authorization header")
        
        effective_api_key = token_api_key if token_api_key else api_key
        
        if effective_api_key is None:
            self.logger.warning("No API key provided in either X-API-Key or Authorization header")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        api_key_data = self.api_key_repo.get_api_key_by_value(effective_api_key)
        
        if api_key_data is None:
            self.logger.warning(f"Invalid API key: {effective_api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check the API key's active status
        if not api_key_data["is_active"]:
            self.logger.warning(f"Inactive API key: {effective_api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is inactive",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Check API key expiration date
        if api_key_data["expiry_date"]:
            expiry_date = api_key_data["expiry_date"]
            expiry_date_aware = expiry_date.replace(tzinfo=timezone.utc) if expiry_date.tzinfo is None else expiry_date
            if expiry_date_aware < datetime.now(timezone.utc):
                self.logger.warning(f"Expired API key: {effective_api_key[:10]}...")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        user_id = api_key_data["user_id"]
        
        # Get user info in database (mysql)
        user_info = self.user_role_service.get_user_info_with_roles(user_id)
        
        if not user_info:
            self.logger.warning(f"User {user_id} not found in system")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in system",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Use organization_id from header if present, otherwise use from API key
        effective_org_id = organization_id if organization_id else api_key_data["organization_id"]
        
        # Check and authenticate permissions with the organization if any
        if effective_org_id:
            # Check if organization exists
            if effective_org_id not in [org["organization_id"] for org in user_info.get("organizations", [])]:
                # Check directly if not found in user information
                org_exists = self.user_role_service.verify_organization_exists(effective_org_id)
                if not org_exists:
                    self.logger.warning(f"Organization {effective_org_id} not found in system")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Organization not found in system",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
            
            # Check if user has permissions to organization
            user_role = user_info.get("roles", {}).get(effective_org_id)
            
            has_access = False
            if user_role:
                if require_role == "ADMIN":
                    has_access = user_role == "ADMIN"
                else:
                    has_access = user_role in ["USER", "ADMIN"] 
            
            if not has_access:
                self.logger.warning(f"User {user_id} does not have required access to organization {effective_org_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have required access to organization",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Update last used time and number of uses
        self.api_key_repo.update_api_key_usage(effective_api_key)
        
        # Attach user_id, organization_id and role information to request state so it can be accessed from handlers
        if request:
            request.state.user_id = user_id
            request.state.organization_id = effective_org_id
            
            # Get role
            if effective_org_id:
                role = user_info.get("roles", {}).get(effective_org_id)
                request.state.role = role
                self.logger.debug(f"User {user_id} has role {role} in organization {effective_org_id}")
            
            # Add user information to request state
            request.state.user_info = user_info
                
        api_key_data["effective_organization_id"] = effective_org_id
        # self.logger.info(f"api_key_data: {api_key_data}")
        
        return api_key_data
    # End option tesing

    async def author_with_api_key(
        self, 
        organization_id: str = Depends(ORGANIZATION_ID_HEADER),
        api_key: str = Depends(API_KEY_HEADER),
        request: Request = None,
        require_role: str = None
    ) -> Optional[Dict[str, Any]]:

        if api_key is None:
            self.logger.warning("No API key provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "APIKey"}
            )
            
        # Get API key data as dictionary instead of ORM object
        api_key_data = self.api_key_repo.get_api_key_by_value(api_key)
        
        if api_key_data is None:
            self.logger.warning(f"Invalid API key: {api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "APIKey"}
            )
            
        # Check the API key's active status
        if not api_key_data["is_active"]:
            self.logger.warning(f"Inactive API key: {api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is inactive",
                headers={"WWW-Authenticate": "APIKey"}
            )
            
        # Check API key expiration date
        if api_key_data["expiry_date"]:
            expiry_date = api_key_data["expiry_date"]
            expiry_date_aware = expiry_date.replace(tzinfo=timezone.utc) if expiry_date.tzinfo is None else expiry_date
            if expiry_date_aware < datetime.now(timezone.utc):
                self.logger.warning(f"Expired API key: {api_key[:10]}...")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired",
                    headers={"WWW-Authenticate": "APIKey"}
                )
        
        user_id = api_key_data["user_id"]
        
        # Get user info in database (mysql)
        user_info = self.user_role_service.get_user_info_with_roles(user_id)
        
        if not user_info:
            self.logger.warning(f"User {user_id} not found in system")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in system",
                headers={"WWW-Authenticate": "APIKey"}
            )
        
        # Use organization_id from header if present, otherwise use from API key
        effective_org_id = organization_id if organization_id else api_key_data["organization_id"]
        
         # Check and authenticate permissions with the organization if any
        if effective_org_id:
            # Check if organization exists
            if effective_org_id not in [org["organization_id"] for org in user_info.get("organizations", [])]:
                # Check directly if not found in user information
                org_exists = self.user_role_service.verify_organization_exists(effective_org_id)
                if not org_exists:
                    self.logger.warning(f"Organization {effective_org_id} not found in system")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Organization not found in system",
                        headers={"WWW-Authenticate": "APIKey"}
                    )
            
            # Check if user has permissions to organization
            user_role = user_info.get("roles", {}).get(effective_org_id)
            
            has_access = False
            if user_role:
                if require_role == "ADMIN":
                    has_access = user_role == "ADMIN"
                else:
                    has_access = user_role in ["USER", "ADMIN"] 
            
            if not has_access:
                self.logger.warning(f"User {user_id} does not have required access to organization {effective_org_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have required access to organization",
                    headers={"WWW-Authenticate": "APIKey"}
                )
        
        # Update last used time and number of uses
        self.api_key_repo.update_api_key_usage(api_key)
        
        # Attach user_id, organization_id and role information to request state so it can be accessed from handlers
        if request:
            request.state.user_id = user_id
            request.state.organization_id = effective_org_id
            
            # Get role
            if effective_org_id:
                role = user_info.get("roles", {}).get(effective_org_id)
                request.state.role = role
                self.logger.debug(f"User {user_id} has role {role} in organization {effective_org_id}")
            
            # Add user information to request state
            request.state.user_info = user_info
                
        api_key_data["effective_organization_id"] = effective_org_id
        # self.logger.info(f"api_key_data: {api_key_data}")
        
        return api_key_data
    
    
    async def admin_required(self,
                             organization_id: str = Depends(ORGANIZATION_ID_HEADER),
                             api_key: str = Depends(API_KEY_HEADER),
                             request: Request = None
                            ) -> Optional[Dict[str, Any]]:
        """
        Authentication of request and checking Admin rights

        Args:
            api_key: API key from header
            organization_id: Organization ID from header
            request: Request object

        Returns:
            Optional[Dict[str, Any]]: API key information if authentication is successful and user is Admin
        """
        return await self.author_with_api_key(
            organization_id=organization_id,
            api_key=api_key,
            request=request,
            require_role="ADMIN"
        )


    def generate_api_key(self, length: int = 40) -> str:
        alphabet = string.ascii_letters + string.digits
        api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        return f"think_{api_key}"


    def create_api_key(
        self, 
        user_id: str, 
        organization_id: Optional[str] = None,
        name: Optional[str] = None,
        expires_in_days: int = 365
    ) -> Dict[str, Any]:
        
        user_exists = self.user_role_service.verify_user_exists(user_id)
        if not user_exists:
            raise ValueError(f"User {user_id} not found in system")
        
        # Verify organization and access if organization_id is provided
        if organization_id:
            org_exists = self.user_role_service.verify_organization_exists(organization_id)
            if not org_exists:
                raise ValueError(f"Organization {organization_id} not found in frontend system")
            
            # Check if user has organization access
            has_access = self.user_role_service.verify_access(user_id, organization_id)
            if not has_access:
                raise ValueError(f"User {user_id} does not have access to organization {organization_id}")
                
        api_key = self.generate_api_key()
        expiry_date = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        api_key_id = self.api_key_repo.create_api_key(
            user_id=user_id,
            organization_id=organization_id,
            api_key=api_key,
            name=name,
            expiry_date=expiry_date
        )
        
        # self.logger.info(f"Created API key for user {user_id}")
        
        role = None
        if organization_id:
            role = self.user_role_service.get_user_role(user_id, organization_id)
        
        # Get user info from database
        user_info = self.user_role_service.get_user_info_with_roles(user_id)
        user_name = user_info.get('full_name') if user_info else None
        
        return {
            "id": api_key_id,
            "api_key": api_key,
            "user_id": user_id,
            "user_name": user_name,
            "organization_id": organization_id,
            "name": name,
            "role": role,
            "expiry_date": expiry_date.isoformat(),
            "is_active": True
        }


    def revoke_api_key(self, api_key_id: str, user_id: str = None) -> bool:
        """
        Revoke (deactivate) API key

        Args:
            api_key_id: ID of API key to revoke
            user_id: ID of user (to check ownership)

        Returns:
            bool: True on success, False on failure
        """
        if user_id:
            api_key_data = self.api_key_repo.get_api_key_by_id(api_key_id)
            if not api_key_data or api_key_data["user_id"] != user_id:
                self.logger.warning(f"User {user_id} attempted to revoke API key {api_key_id} they don't own")
                return False
                    
        return self.api_key_repo.deactivate_api_key(api_key_id)


    def delete_api_key(self, api_key_id: str, user_id: str = None) -> bool:
        """
        Delete API key

        Args:
            api_key_id: ID of the API key to delete
            user_id: ID of the user (to check ownership)

        Returns:
            bool: True on success, False on failure
        """
        if user_id:
            api_key_data = self.api_key_repo.get_api_key_by_id(api_key_id)
            if not api_key_data or api_key_data["user_id"] != user_id:
                self.logger.warning(f"User {user_id} attempted to delete API key {api_key_id} they don't own")
                return False
                    
        return self.api_key_repo.delete_api_key(api_key_id)


    def get_user_api_keys(self, user_id: str) -> list:
        """
        Get a list of user API keys

        Args:
            user_id: User ID

        Returns:
            list: List of API keys
        """
        user_exists = self.user_role_service.verify_user_exists(user_id)
        if not user_exists:
            self.logger.warning(f"User {user_id} not found in system")
            return []
        
        # List API key from repository
        keys = self.api_key_repo.get_api_keys_by_user(user_id)
        
        # Get user info with all organizations and roles
        user_info = self.user_role_service.get_user_info_with_roles(user_id)
        
        for key in keys:
            org_id = key.get("organization_id")
            if org_id and user_info and "roles" in user_info:
                # Use roles from user_info instead of separate query
                key["role"] = user_info["roles"].get(org_id)
                
                # Check if organization exists
                org_exists = any(org["organization_id"] == org_id for org in user_info.get("organizations", []))
                key["organization_exists"] = org_exists
            
            if user_info:
                key["user_name"] = user_info.get("full_name")
                key["user_email"] = user_info.get("email")
                
        return keys
        

    def get_user_organizations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get the user's organization list

        Args:
            user_id: User ID

        Returns:
            List[Dict[str, Any]]: Organization list
        """
        user_exists = self.user_role_service.verify_user_exists(user_id)
        if not user_exists:
            self.logger.warning(f"User {user_id} not found in system")
            return []
        
        return self.user_role_service.get_user_organizations(user_id)
    

    def cache_manager(self, user_id: str, organization_id: str):
        return self.user_role_service.clear_cache(user_id, organization_id)