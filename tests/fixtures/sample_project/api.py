"""API endpoints that use auth and database modules."""

from auth import validate_token, generate_token
from database import get_user, save_user


def login(user_id: int, token: str) -> dict:
    """
    Handle user login.
    
    Args:
        user_id: The user's ID.
        token: Authentication token.
        
    Returns:
        Login response dictionary.
    """
    if not validate_token(token):
        return {"error": "Invalid token"}
    
    user = get_user(user_id)
    return {"success": True, "user": user}


def register(user_id: int) -> dict:
    """
    Register a new user.
    
    Args:
        user_id: The new user's ID.
        
    Returns:
        Registration response with new token.
    """
    user_data = {"id": user_id, "name": f"NewUser{user_id}"}
    save_user(user_data)
    new_token = generate_token(user_id)
    return {"success": True, "token": new_token}
