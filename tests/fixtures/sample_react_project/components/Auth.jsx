import React, { useState } from 'react';
import { fetchLogin } from '../api/auth';

/**
 * Authentication component for user login
 * Handles login form and authentication flow
 */
const Auth = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  /**
   * Handle login form submission
   * Calls the login API and processes the response
   */
  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const result = await fetchLogin(email, password);
      if (result.success) {
        console.log('Login successful');
      }
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleLogin}>
      <input 
        type="email" 
        value={email} 
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
      />
      <input 
        type="password" 
        value={password} 
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};

export default Auth;
