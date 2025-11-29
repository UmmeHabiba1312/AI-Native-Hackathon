import React, { createContext, useContext, useState, useEffect } from 'react';

// 1. Create Context
const SessionContext = createContext({
  data: { user: { name: "Judge", hasGPU: undefined } },
  error: null
});

// 2. Export Provider
export const SessionProvider = ({ children, client }) => {
  return (
    <SessionContext.Provider value={{ data: { user: { name: "Judge", hasGPU: undefined } }, error: null }}>
      {children}
    </SessionContext.Provider>
  );
};

// 3. Export Hook
export const useSession = () => {
  return useContext(SessionContext);
};

// 4. Export Client Helper (FIXED STRUCTURE)
export const createAuthClient = (config) => ({
  signIn: {
    email: async ({ email, password }) => {
      console.log(`Mock Signing in ${email}...`);
      return { data: { token: "mock-token" }, error: null };
    }
  },
  signOut: async () => console.log("Mock Sign Out")
});