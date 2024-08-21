import React, { useState } from 'react';
import Login from './Login';
import Signup from './Signup';
import Chat from './Chat';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isSignup, setIsSignup] = useState(false);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  const handleSignup = () => {
    setIsSignup(false);
    setIsLoggedIn(true);
  };

  return (
    <div>
      {!isLoggedIn ? (
        isSignup ? (
          <Signup onSignup={handleSignup} />
        ) : (
          <Login onLogin={handleLogin} />
        )
      ) : (
        <Chat />
      )}
      {!isLoggedIn && (
        <button onClick={() => setIsSignup(!isSignup)} className="absolute top-4 right-4 bg-blue-500 text-white p-2 rounded">
          {isSignup ? 'Login' : 'Signup'}
        </button>
      )}
    </div>
  );
}

export default App;
