import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useHistory } from '@docusaurus/router';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [isLogin, setIsLogin] = useState(true);
  const history = useHistory();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const mode = isLogin ? 'login' : 'signup';
    const body = isLogin ? { email, password } : { email, password, name };

    try {
      const response = await fetch(`https://ai-native-hackathon-backend.vercel.app/auth/${mode}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (data.status === 'success') {
        localStorage.setItem('user', JSON.stringify(data.user));
        if (!isLogin) {
          localStorage.removeItem('has_seen_onboarding');
        }
        history.push('/');
        window.location.reload();
      } else {
        alert(data.message || 'Authentication failed');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      alert('An error occurred during authentication.');
    }
  };

  return (
    <Layout title="Auth">
      <div style={{
        display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh',
        flexDirection: 'column'
      }}>
        <h1>{isLogin ? 'Student Login' : 'Student Sign Up'}</h1>
        <button
          onClick={() => setIsLogin(!isLogin)}
          style={{ marginBottom: '1rem', padding: '8px 16px', borderRadius: '5px', border: '1px solid #ccc', cursor: 'pointer' }}
        >
          Switch to {isLogin ? 'Sign Up' : 'Login'}
        </button>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '300px' }}>
          {!isLogin && (
            <input
              type="text"
              placeholder="Enter Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
              required
            />
          )}
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
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
            {isLogin ? 'Sign In' : 'Sign Up'}
          </button>
        </form>
      </div>
    </Layout>
  );
}