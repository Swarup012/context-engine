// User management functions

interface User {
  id: number;
  name: string;
  email: string;
}

function createUser(name: string, email: string): User {
  const isValid = validateEmail(email);
  if (!isValid) {
    throw new Error('Invalid email');
  }
  
  return {
    id: Date.now(),
    name: name,
    email: email
  };
}

const getUserById = async (id: number): Promise<User | null> => {
  // Simulated database call
  return null;
};

export { createUser, getUserById };
