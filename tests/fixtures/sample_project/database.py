"""Database operations module."""


def get_user(user_id: int) -> dict:
    """
    Retrieve a user from the database.
    
    Args:
        user_id: The ID of the user to retrieve.
        
    Returns:
        A dictionary containing user data.
    """
    # Simulate database lookup
    return {"id": user_id, "name": f"User{user_id}"}


def save_user(user_data: dict) -> bool:
    """
    Save user data to the database.
    
    Args:
        user_data: Dictionary containing user information.
        
    Returns:
        True if save was successful.
    """
    # Simulate database save
    return True
