import React, { useState } from 'react';

function Chat() {
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessions] = useState(['Chat 1', 'Chat 2', 'Chat 3']);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const selectSession = (session) => {
    setSelectedSession(session);
    // Placeholder for loading messages from the selected session
    setMessages([
      { text: 'Hello!', fromUser: false },
      { text: 'Hi, how are you?', fromUser: true },
    ]);
  };

  const sendMessage = () => {
    if (input.trim() !== '') {
      setMessages([...messages, { text: input, fromUser: true }]);
      setInput('');
    }
  };

  return (
    <div className="flex h-screen">
      <div className="w-1/4 bg-gray-100 p-4">
        <h3 className="text-xl font-bold mb-4">Chat Sessions</h3>
        <ul>
          {sessions.map((session, index) => (
            <li
              key={index}
              onClick={() => selectSession(session)}
              className="cursor-pointer p-2 hover:bg-gray-200"
            >
              {session}
            </li>
          ))}
        </ul>
      </div>
      <div className="w-2/3 p-4 flex flex-col">
        {selectedSession ? (
          <>
            <h3 className="text-xl font-bold mb-4">{selectedSession}</h3>
            <div className="flex-1 overflow-y-auto">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-2 p-2 rounded ${message.fromUser ? 'bg-blue-100 text-right' : 'bg-gray-200 text-left'}`}
                >
                  {message.text}
                </div>
              ))}
            </div>
            <div className="mt-4 flex">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a message"
                className="flex-1 p-2 border rounded"
              />
              <button onClick={sendMessage} className="bg-blue-500 text-white p-2 rounded ml-2">Send</button>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <h3 className="text-xl font-bold">Welcome! Select a chat session to start.</h3>
          </div>
        )}
      </div>
    </div>
  );
}

export default Chat;
