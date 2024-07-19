import React, { useState, useEffect } from 'react';
import { User, Bot, HelpCircle, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import conversationData from './conversation2.json';

// Utility function to handle line breaks
const formatText = (text) => {
  return text.split('\n').map((line, index) => (
    <React.Fragment key={index}>
      {line}
      {index < text.split('\n').length - 1 && <br />}
    </React.Fragment>
  ));
};

const Message = ({ message, isSelected, onSelect }) => (
  <div 
    className={`p-4 rounded-lg mb-4 ${
      message.role === 'assistant' ? 'bg-blue-100' : 'bg-gray-100'
    } ${isSelected ? 'ring-2 ring-blue-500' : ''}`}
  >
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center">
        {message.role === 'assistant' ? (
          <Bot className="mr-2 text-blue-500" />
        ) : (
          <User className="mr-2 text-green-500" />
        )}
        <span className="font-bold">{message.role.charAt(0).toUpperCase() + message.role.slice(1)}</span>
      </div>
      {message.role === 'user' && 'score' in message && (
        <button 
          onClick={onSelect}
          className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors duration-200 flex items-center"
        >
          <Info className="mr-1" size={14} />
          View Details
        </button>
      )}
    </div>
    <p className="mb-2">{formatText(message.content)}</p>
  </div>
);

const Details = ({ message }) => (
  <div className="bg-white p-4 rounded-lg shadow-lg">
    <h2 className="text-xl font-bold mb-4">Prompt Details</h2>
    <p className="mb-2"><strong>Score:</strong> {message.score}</p>
    <div className="mb-4">
      <h3 className="font-bold text-green-600 mb-2">Correctly Answered Questions:</h3>
      <ul className="list-disc pl-5">
        {message.correctly_answered_qs.map((q, i) => (
          <li key={i} className="mb-1 flex items-start">
            <HelpCircle className="mr-2 mt-1 text-green-500 flex-shrink-0" />
            <span>{formatText(q)}</span>
          </li>
        ))}
      </ul>
    </div>
    <div>
      <h3 className="font-bold text-red-600 mb-2">Wrongly Answered Questions:</h3>
      <ul className="list-disc pl-5">
        {message.wrongly_answered_qs.map((q, i) => (
          <li key={i} className="mb-1 flex items-start">
            <HelpCircle className="mr-2 mt-1 text-red-500 flex-shrink-0" />
            <span>{formatText(q)}</span>
          </li>
        ))}
      </ul>
    </div>
  </div>
);

const ScoreChart = ({ conversation }) => {
  const data = conversation
    .filter(msg => msg.role === 'user' && 'score' in msg)
    .map((msg, index) => ({ name: `Prompt ${index + 1}`, score: msg.score }));

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg mb-4">
      <h2 className="text-xl font-bold mb-4">Score Over Time</h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="score" stroke="#8884d8" activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

function App() {
  const [conversation, setConversation] = useState([]);
  const [selectedMessage, setSelectedMessage] = useState(null);

  useEffect(() => {
    setConversation(conversationData);
  }, []);

  return (
    <div className="App bg-gray-100 h-screen flex">
      <div className="flex flex-row w-full">
        <div className="w-1/2 p-4 overflow-y-auto bg-white">
          {conversation.map((message, index) => (
            <Message 
              key={index} 
              message={message} 
              isSelected={selectedMessage === index}
              onSelect={() => setSelectedMessage(index)}
            />
          ))}
        </div>
        <div className="w-1/2 p-4 bg-gray-50 overflow-y-auto flex flex-col">
          <ScoreChart conversation={conversation} />
          {selectedMessage !== null && 'score' in conversation[selectedMessage] ? (
            <Details message={conversation[selectedMessage]} />
          ) : (
            <div className="flex items-center justify-center flex-grow text-gray-500">
              Select a user message to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;