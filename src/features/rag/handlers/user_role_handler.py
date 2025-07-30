import time
from typing import Optional, Dict, List, Any, Tuple
from src.core.utils.logger.custom_logging import LoggerMixin
from src.core.database.mysql_connection import mysql_connection


class UserRoleService(LoggerMixin):    
    def __init__(self):
        super().__init__()
        self.db = mysql_connection
        self._cache_ttl = 300  # Cache Expiration Time (5 minutes)
        self._role_cache = {}  # Role cache: {(user_id, org_id): (role, timestamp)}
        self._user_cache = {}  # Cache user information: {user_id: (user_info, timestamp)}
        self._org_cache = {}   # Cache exists organization: {org_id: (exists, timestamp)}
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache is still valid"""
        return (time.time() - timestamp) < self._cache_ttl
    
    def get_user_info_with_roles(self, user_id: str) -> Dict[str, Any]:
        """
        Get user and role information in a query

        Args:
            user_id: ID of the user

        Returns:
            Dict[str, Any]: User and role information
        """
        # Check cache
        if user_id in self._user_cache:
            cached_info, timestamp = self._user_cache[user_id]
            if self._is_cache_valid(timestamp):
                return cached_info
        
        try:
            user_query = """
                SELECT 
                    u.Id, u.Code, u.Email, u.Firstname, u.Lastname, 
                    u.Phone, u.Gender, u.Avatar, u.DefaultOrganizationId
                FROM 
                    User u
                WHERE 
                    u.Id = %s AND u.Status = 10
            """
            
            user_result = self.db.execute_query(user_query, (user_id,))
            
            if not user_result:
                return None
            
            user_info = user_result[0]
    
            role_query = """
                SELECT 
                    ou.OrganizationId, 
                    o.Name as organization_name, 
                    o.Code as organization_code,
                    ou.Role as role_id
                FROM 
                    OrganizationUser ou
                JOIN 
                    Organization o ON ou.OrganizationId = o.Id
                WHERE 
                    ou.UserId = %s AND ou.Status = 10 AND o.Status = 10
            """
            
            role_results = self.db.execute_query(role_query, (user_id,))
            
            roles = {}
            organizations = []
            for row in role_results:
                org_id = str(row['OrganizationId'])
                role_id = row['role_id']
                
                # Convert role_id to role string
                if role_id == 10:
                    role = "ADMIN"
                elif role_id == 90:
                    role = "USER"
                else:
                    role = f"ROLE_{role_id}"
                
                roles[org_id] = role
                
                # Save to role cache
                self._role_cache[(user_id, org_id)] = (role, time.time())
                
                # Add to list org
                organizations.append({
                    "organization_id": org_id,
                    "name": row['organization_name'],
                    "code": row['organization_code'],
                    "role": role
                })
                
                # Mark organization as existing
                self._org_cache[org_id] = (True, time.time())
            
            result = {
                "user_id": str(user_info['Id']),
                "code": user_info['Code'],
                "email": user_info['Email'],
                "first_name": user_info['Firstname'],
                "last_name": user_info['Lastname'],
                "full_name": f"{user_info['Firstname']} {user_info['Lastname']}".strip(),
                "phone": user_info['Phone'],
                "gender": user_info['Gender'],
                "avatar": user_info['Avatar'],
                "default_organization_id": str(user_info['DefaultOrganizationId']) if user_info['DefaultOrganizationId'] else None,
                "organizations": organizations,
                "roles": roles,
                "exists": True
            }
            
            # Save to cache
            self._user_cache[user_id] = (result, time.time())
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error getting user info with roles: {str(e)}")
            return None
    
    def get_user_role(self, user_id: str, organization_id: str) -> Optional[str]:
        """
        Get the user role in the organization, cache preferred

        Args:
            user_id: User ID
            organization_id: Organization ID

        Returns:
            Optional[str]: Role or None if not found
        """
        # Check cache
        cache_key = (user_id, organization_id)
        if cache_key in self._role_cache:
            role, timestamp = self._role_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return role
        
        user_info = self.get_user_info_with_roles(user_id)
        if user_info and "roles" in user_info:
            if organization_id in user_info["roles"]:
                return user_info["roles"][organization_id]
            
            return None
            
        # Query directly if not in cache
        try:
            query = """
                SELECT Role 
                FROM OrganizationUser 
                WHERE UserId = %s AND OrganizationId = %s AND Status = 10
            """
            
            result = self.db.execute_query(query, (user_id, organization_id))
            
            if not result:
                self.logger.warning(f"No role found for user {user_id} in organization {organization_id}")
                self._role_cache[cache_key] = (None, time.time())
                return None
                
            # 10 = ADMIN, 90 = USER | Admin = 10, Owner = 75, Member = 90,
            role_id = result[0]['Role']
            
            if role_id == 10:
                role = "ADMIN"
            elif role_id == 90:
                role = "USER"
            else:
                role = f"ROLE_{role_id}"
                
            self._role_cache[cache_key] = (role, time.time())
            
            return role
                
        except Exception as e:
            self.logger.error(f"Error getting user role: {str(e)}")
            return None
    
    def verify_access(self, user_id: str, organization_id: str, required_role: str = "USER") -> bool:
        """
        Check access, cache preferred

        Args:
            user_id: User ID
            organization_id: Organization ID
            required_role: Required role (default: "USER")

        Returns:
            bool: True if permission is present, False otherwise
        """
        role = self.get_user_role(user_id, organization_id)
        
        if role is None:
            return False
            
        if required_role == "ADMIN":
            return role == "ADMIN"
        else:
            return role in ["USER", "ADMIN"]
            
    def verify_user_exists(self, user_id: str) -> bool:
        """
        Check if user exists, cache preferred

        Args:
            user_id: User ID

        Returns:
            bool: True if exists, False otherwise
        """
        if user_id in self._user_cache:
            user_info, timestamp = self._user_cache[user_id]
            if self._is_cache_valid(timestamp):
                return user_info.get("exists", False)
        
        # Get info
        user_info = self.get_user_info_with_roles(user_id)
        if user_info:
            return True
        
        try:
            query = "SELECT COUNT(*) as count FROM User WHERE Id = %s AND Status = 10"
            result = self.db.execute_query(query, (user_id,))
            exists = result[0]['count'] > 0

            self._user_cache[user_id] = ({"exists": exists}, time.time())
            
            return exists
                
        except Exception as e:
            self.logger.error(f"Error verifying user exists: {str(e)}")
            return False
    
    def verify_organization_exists(self, organization_id: str) -> bool:
        """
        Check if organization exists, cache preferred

        Args:
            organization_id: Organization ID

        Returns:
            bool: True if exists, False otherwise
        """
        if organization_id in self._org_cache:
            exists, timestamp = self._org_cache[organization_id]
            if self._is_cache_valid(timestamp):
                return exists
        
        try:
            query = "SELECT COUNT(*) as count FROM Organization WHERE Id = %s AND Status = 10"
            result = self.db.execute_query(query, (organization_id,))
            
            exists = result[0]['count'] > 0
            
            self._org_cache[organization_id] = (exists, time.time())
            
            return exists
                
        except Exception as e:
            self.logger.error(f"Error verifying organization exists: {str(e)}")
            return False
    
    def get_user_organizations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get the user's organization list, cache preferred

        Args:
            user_id: User ID

        Returns:
            List[Dict[str, Any]]: Organization list
        """
        user_info = self.get_user_info_with_roles(user_id)
        if user_info:
            return user_info.get("organizations", [])
        
        try:
            query = """
                SELECT 
                    ou.OrganizationId, 
                    o.Name as organization_name, 
                    o.Code as organization_code,
                    ou.Role as role_id
                FROM 
                    OrganizationUser ou
                JOIN 
                    Organization o ON ou.OrganizationId = o.Id
                WHERE 
                    ou.UserId = %s AND ou.Status = 10 AND o.Status = 10
            """
            
            results = self.db.execute_query(query, (user_id,))
            
            organizations = []
            for row in results:
                org_id = str(row['OrganizationId'])
                role_id = row['role_id']
                
                if role_id == 10:
                    role = "ADMIN"
                elif role_id == 90:
                    role = "USER"
                else:
                    role = f"ROLE_{role_id}"
                
                self._role_cache[(user_id, org_id)] = (role, time.time())
                
                organizations.append({
                    "organization_id": org_id,
                    "name": row['organization_name'],
                    "code": row['organization_code'],
                    "role": role
                })
                
            return organizations
                
        except Exception as e:
            self.logger.error(f"Error getting user organizations: {str(e)}")
            return []
    

    def is_admin(self, user_id: str, organization_id: str) -> bool:
        role = self.get_user_role(user_id, organization_id)
        return role == "ADMIN"
    

    def clear_cache(self, user_id: Optional[str] = None, organization_id: Optional[str] = None):
        """
        Clear cache, can choose to clear by user_id or organization_id

        Args:
            user_id: User ID (if you want to clear for a specific user)
            organization_id: Organization ID (if you want to clear for a specific organization)
        """
        if user_id and organization_id:
            # Clear cache specific to user and organization
            if (user_id, organization_id) in self._role_cache:
                del self._role_cache[(user_id, organization_id)]
        elif user_id:
            if user_id in self._user_cache:
                del self._user_cache[user_id]
            
            keys_to_delete = [k for k in self._role_cache.keys() if k[0] == user_id]
            for key in keys_to_delete:
                del self._role_cache[key]

        elif organization_id:
            if organization_id in self._org_cache:
                del self._org_cache[organization_id]
            
            keys_to_delete = [k for k in self._role_cache.keys() if k[1] == organization_id]
            for key in keys_to_delete:
                del self._role_cache[key]
        else:
            self._role_cache.clear()
            self._user_cache.clear()
            self._org_cache.clear()