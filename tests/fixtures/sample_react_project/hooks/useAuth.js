import { useState, useEffect } from 'react';
import { fetchLogin, fetchUserData, logout } from '../api/auth';

/**
 * Custom authentication hook
 * Manages authentication state and provides auth functions
 * @returns {Object} Auth state and functions
 */
export function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Initialize auth state on mount
   */
  useEffect(() => {
    checkAuth();
  }, []);

  /**
   * Check if user is authenticated
   * Loads user data if token exists
   */
  const checkAuth = async () => {
    try {
      const userData = await fetchUserData();
      setUser(userData);
    } catch (err) {
      setError(err);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Login with email and password
   * @param {string} email - User email
   * @param {string} password - User password
   */
  const login = async (email, password) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await fetchLogin(email, password);
      
      if (result.token) {
        localStorage.setItem('authToken', result.token);
        await checkAuth();
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Logout current user
   */
  const handleLogout = () => {
    logout();
    setUser(null);
  };

  return {
    user,
    loading,
    error,
    login,
    logout: handleLogout,
    isAuthenticated: !!user
  };
}
