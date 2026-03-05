import React, { useEffect, useState } from 'react';
import Auth from './Auth';
import { fetchUserData } from '../api/auth';

/**
 * Dashboard component
 * Main application dashboard that shows user data
 */
const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  /**
   * Load user data on component mount
   */
  useEffect(() => {
    loadUserData();
  }, []);

  /**
   * Fetch and set user data
   */
  const loadUserData = async () => {
    try {
      const userData = await fetchUserData();
      setUser(userData);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Failed to load user data:', error);
      setIsAuthenticated(false);
    }
  };

  if (!isAuthenticated) {
    return <Auth />;
  }

  return (
    <div className="dashboard">
      <h1>Welcome, {user?.name}</h1>
      <p>Email: {user?.email}</p>
    </div>
  );
};

export default Dashboard;
