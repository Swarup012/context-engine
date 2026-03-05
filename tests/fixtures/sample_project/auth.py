"""Authentication module with token validation."""


def validate_token(token: str) -> bool:
    """
    Validate an authentication token.
    
    Args:
        token: The token to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not token:
        return False
    return check_token_format(token)


def check_token_format(token: str) -> bool:
    """Check if token has correct format."""
    return len(token) >= 32 and token.startswith("tk_")


def generate_token(user_id: int) -> str:
    """
    Generate a new authentication token for a user.
    
    Args:
        user_id: The ID of the user.
        
    Returns:
        A new authentication token.
    """
    return f"tk_{user_id}_{'x' * 30}"
