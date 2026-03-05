// API component for React

import React from 'react';

const ApiComponent = () => {
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const user = createUser("John", "john@example.com");
    console.log(user);
  };

  return (
    <form onSubmit={handleSubmit}>
      <button type="submit">Submit</button>
    </form>
  );
};

export default ApiComponent;
