import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useHistory } from '@docusaurus/router';
import authClient from "../lib/auth-client"; // Fixed Import

export default function Login() {
  const [email, setEmail] = useState('');
  const history = useHistory();

  const handleLogin = async (e) => {
    e.preventDefault();
    // Use the client instance
    await authClient.signIn.email({ email, password: "password" });
    alert("Logged in successfully! (Hackathon Mock)");
    localStorage.setItem("user_session", "active");
    history.push('/'); // Redirect to Home
    window.location.reload(); // Force refresh to trigger Onboarding Modal
  };

  return (
    <Layout title="Login">
      <div style={{
        display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh',
        flexDirection: 'column'
      }}>
        <h1>Student Login</h1>
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '300px' }}>
          <input
            type="email"
            placeholder="Enter Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
            required
          />
          <input
            type="password"
            placeholder="Password"
            style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
            required
          />
          <button
            type="submit"
            style={{
              padding: '10px', backgroundColor: '#2e8555', color: 'white',
              border: 'none', borderRadius: '5px', cursor: 'pointer', fontSize: '1rem'
            }}
          >
            Sign In
          </button>
        </form>
      </div>
    </Layout>
  );
}