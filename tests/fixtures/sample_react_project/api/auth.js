/**
 * Authentication API module
 * Handles all authentication-related API calls
 */

const API_BASE = 'https://api.example.com';

/**
 * Login user with email and password
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise<Object>} Login result with token
 */
export async function fetchLogin(email, password) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  
  return await response.json();
}

/**
 * Fetch current user data
 * Requires authentication token
 * @returns {Promise<Object>} User data object
 */
export async function fetchUserData() {
  const token = localStorage.getItem('authToken');
  
  const response = await fetch(`${API_BASE}/user/me`, {
    headers: { 
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  });
  
  return await response.json();
}

/**
 * Logout current user
 * Clears authentication token
 */
export function logout() {
  localStorage.removeItem('authToken');
}
