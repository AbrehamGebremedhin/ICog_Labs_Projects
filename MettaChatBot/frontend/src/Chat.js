import React, { useState, useEffect } from 'react';
import axios from 'axios';

function Chat() {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const getCookieValue = (cookieName) => {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.startsWith(cookieName + '=')) {
        return cookie.substring(cookieName.length + 1);
      }
    }
    return null; // Cookie not found
  };

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/v1/sessions/', {
          headers: {
            'Authorization': `Bearer ${getCookieValue("token")}`
          }
        });
        setSessions(response.data);
      } catch (error) {
        console.error('Error fetching sessions:', error);
      }
    };

    fetchSessions();
  }, []);

  const fetchMessages = async (session_id) => {
    try {
      setSelectedSession(session_id);
      const response = await axios.get(`http://127.0.0.1:8000/api/v1/sessions/${session_id}`, {
        headers: {
          'Authorization': `Bearer ${getCookieValue("token")}`
        }
      });
      console.log({ response: response.data });
      setMessages(response.data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async () => {
    if (selectedSession === null) { 
      const response = await fetch("http://127.0.0.1:8000/api/v1/sessions/", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getCookieValue('token')}`
        },
        body: JSON.stringify({ session_name: input })
      });

      const data = await response.json();
      setSelectedSession(data.id);
      return;
    }
    
    const newMessage = {
      "content": input,
      "sender_type": "USER",
      "session_name": input,
      "session": selectedSession
    };

    setInput('');
    setMessages([...messages, newMessage]);
    
    try {
      const response = await fetch("http://127.0.0.1:8000/api/v1/messages/", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getCookieValue('token')}`
        },
        body: JSON.stringify(newMessage)
      });

      await response.json();
      fetchMessages(selectedSession);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="flex h-screen">
      <div className="w-1/4 bg-gray-100 p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold">Chat Sessions</h3>
          <button
            onClick={() => { setSelectedSession(null); setMessages([]); }}
            className="bg-blue-500 text-white p-2 rounded"
          >
            Reset
          </button>
        </div>
        <ul>
          {sessions.map((session) => (
            <li
              key={session.id}
              onClick={() => fetchMessages(session.id)}
              className={`cursor-pointer p-2 ${selectedSession?.id === session.id ? 'bg-blue-200' : 'hover:bg-gray-200'}`}
            >
              {session.session_name}
            </li>
          ))}
        </ul>
      </div>
      <div className="w-2/3 p-4 flex flex-col">
        {selectedSession ? (
          <>
            <h3 className="text-xl font-bold mb-4">{selectedSession.session_name}</h3>
            <div className="flex-1 overflow-y-auto">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-2 p-2 rounded ${message.sender_type === "USER" ? 'bg-blue-100 text-right' : 'bg-gray-200 text-left'}`}
                >
                  {message.content}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <h3 className="text-xl font-bold">Welcome! Select a chat session to start or send a message to create a new session.</h3>
          </div>
        )}
        <div className="mt-4 flex">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message"
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={sendMessage}
            className="bg-blue-500 text-white p-2 rounded ml-2"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat;
