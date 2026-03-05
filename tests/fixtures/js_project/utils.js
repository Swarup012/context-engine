// Utility functions for the application

function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function formatUserName(firstName, lastName) {
  return `${firstName} ${lastName}`;
}

const calculateTotal = (items) => {
  return items.reduce((sum, item) => sum + item.price, 0);
};

export { validateEmail, formatUserName, calculateTotal };
