import React, { useState, useRef, useEffect } from 'react';
import { Bot, MessageSquare, X, Send, RotateCcw, Sparkles, Eye, Car, ChevronDown, FolderKanban } from 'lucide-react';
import { DocumentManager } from './DocumentManager';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: Array<{ source: string; page: number }>;
}

export interface DevelMoChatWidgetProps {
  apiEndpoint?: string;
  sessionId?: string;
  initialMessage?: string;
  primaryColor?: string;
}

export const DevelMoChatWidget: React.FC<DevelMoChatWidgetProps> = ({
  apiEndpoint = 'http://localhost:8000/api/chat',
  sessionId = 'react_user_session',
  initialMessage = 'Hello! 👋 Welcome to **DevelMo**. How can I help you scale your business with custom AI, computer vision, or software today?',
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat');
  const [inputQuery, setInputQuery] = useState<string>('');
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: initialMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [showQuickReplies, setShowQuickReplies] = useState<boolean>(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (activeTab === 'chat') {
      scrollToBottom();
    }
  }, [messages, isTyping, activeTab]);

  const handleSend = async (queryText?: string) => {
    const text = queryText || inputQuery;
    if (!text.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev: Message[]) => [...prev, userMsg]);
    setInputQuery('');
    setShowQuickReplies(false);
    setIsTyping(true);

    const apiBase = apiEndpoint.replace(/\/api\/chat\/?$/, '');
    const endpoints = Array.from(
      new Set([apiEndpoint, `${apiBase}/api/chat`, 'http://localhost:8000/api/chat', '/api/chat'])
    );

    for (const ep of endpoints) {
      try {
        const response = await fetch(ep, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: text, session_id: sessionId }),
        });

        if (response.ok) {
          const data = await response.json();
          setIsTyping(false);
          setMessages((prev: Message[]) => [
            ...prev,
            {
              id: (Date.now() + 1).toString(),
              role: 'assistant',
              content: data.response || 'No response returned.',
              sources: data.sources,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            },
          ]);
          return;
        }
      } catch (err) {
        // Try next endpoint
      }
    }

    setIsTyping(false);
    setMessages((prev: Message[]) => [
      ...prev,
      {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '⚠️ Could not connect to DevelMo RAG server. Please ensure FastAPI backend is running on http://localhost:8000',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };


  const handleClearMemory = () => {
    setMessages([
      {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Conversation memory reset! How can I assist you now?',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  const apiBaseUrl = apiEndpoint.replace(/\/api\/chat\/?$/, '');

  return (
    <>
      {/* Floating Action Button (FAB) */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle DevelMo AI Chatbot"
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-[#00E5FF] hover:bg-[#00B8CC] text-[#070A11] flex items-center justify-center shadow-[0_0_25px_-5px_rgba(0,229,255,0.5)] transition-all transform hover:scale-105 active:scale-95 group"
      >
        <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 border-2 border-[#070A11] animate-pulse" />
        {isOpen ? (
          <ChevronDown className="w-6 h-6 stroke-[2.5]" />
        ) : (
          <MessageSquare className="w-6 h-6 stroke-[2.5] group-hover:rotate-6 transition-transform" />
        )}
      </button>

      {/* Chat / Document Window Container */}
      <div
        className={`fixed bottom-24 right-6 z-50 transition-all duration-300 transform origin-bottom-right rounded-3xl bg-[#0F1623] border border-[#1E293B] shadow-[0_20px_50px_-10px_rgba(0,0,0,0.5),0_0_30px_rgba(0,229,255,0.15)] flex flex-col overflow-hidden ${
          activeTab === 'documents' ? 'w-[92vw] sm:w-[760px] h-[640px] max-h-[85vh]' : 'w-[90vw] sm:w-[400px] h-[580px] max-h-[82vh]'
        } ${isOpen ? 'scale-100 opacity-100 pointer-events-auto' : 'scale-90 opacity-0 pointer-events-none'}`}
      >
        {/* Header */}
        <div className="bg-[#070A11] px-5 py-3.5 border-b border-[#1E293B] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-cyan-400 to-blue-600 p-[2px]">
                <div className="w-full h-full bg-[#070A11] rounded-full flex items-center justify-center">
                  <Bot className="w-5 h-5 text-[#00E5FF]" />
                </div>
              </div>
              <span className="absolute bottom-0 right-0 w-3 h-3 rounded-full bg-emerald-500 border-2 border-[#070A11]" />
            </div>
            <div>
              <h3 className="font-sans font-bold text-white text-base leading-tight flex items-center gap-1.5">
                DevelMo AI
                <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-cyan-950 text-[#00E5FF] border border-cyan-500/30">
                  RAG 2.0
                </span>
              </h3>
              <p className="text-xs text-[#94A3B8] flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Online & Ready
              </p>
            </div>
          </div>

          {/* Navigation Tabs & Header Controls */}
          <div className="flex items-center gap-1">
            <div className="flex bg-[#0F1623] p-1 rounded-xl border border-[#1E293B] mr-1">
              <button
                onClick={() => setActiveTab('chat')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  activeTab === 'chat'
                    ? 'bg-[#00E5FF] text-[#070A11] font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Chat
              </button>
              <button
                onClick={() => setActiveTab('documents')}
                className={`px-3 py-1 rounded-lg text-xs font-medium flex items-center gap-1 transition-all ${
                  activeTab === 'documents'
                    ? 'bg-[#00E5FF] text-[#070A11] font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <FolderKanban className="w-3.5 h-3.5" />
                Docs
              </button>
            </div>

            {activeTab === 'chat' && (
              <button
                onClick={handleClearMemory}
                title="Reset Memory"
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => setIsOpen(false)}
              title="Close"
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'documents' ? (
          <div className="flex-1 p-4 bg-[#0B0F19] overflow-y-auto">
            <DocumentManager apiBaseUrl={apiBaseUrl} />
          </div>
        ) : (
          <>
            {/* Message Scrollable Area */}
            <div className="flex-1 p-4 overflow-y-auto bg-[#0B0F19] space-y-4 font-sans text-sm">
              {messages.map((msg: Message) => {
                const isUser = msg.role === 'user';
                return (
                  <div key={msg.id} className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}>
                    {!isUser && (
                      <div className="w-8 h-8 rounded-full bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center shrink-0 mt-0.5 text-[#00E5FF]">
                        <Sparkles className="w-4 h-4" />
                      </div>
                    )}

                    <div className={`flex flex-col gap-1 max-w-[82%] ${isUser ? 'items-end' : ''}`}>
                      <span className="text-[11px] text-slate-400 font-medium px-1">
                        {isUser ? 'You' : 'DevelMo AI'} • {msg.timestamp}
                      </span>
                      <div
                        className={`p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm ${
                          isUser
                            ? 'rounded-tr-none bg-[#00E5FF] text-[#070A11] font-medium'
                            : 'rounded-tl-none bg-[#0F1623] border border-[#1E293B] text-slate-200'
                        }`}
                      >
                        {msg.content}

                        {!isUser && msg.sources && msg.sources.length > 0 && (
                          <div className="mt-2.5 pt-2 border-t border-[#1E293B] flex flex-wrap gap-1.5">
                            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block w-full">Sources Cited:</span>
                            {Array.from(new Set(msg.sources.map(s => `${s.source} (Page ${s.page || 1})`))).map((src, i) => (
                              <span key={i} className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-mono">
                                📄 {src}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}


              {/* Quick Reply Chips */}
              {showQuickReplies && (
                <div className="pt-1 flex flex-wrap gap-2">
                  <button
                    onClick={() => handleSend('What flagship products does DevelMo build?')}
                    className="px-3 py-1.5 rounded-full bg-[#0F1623] hover:bg-slate-800 border border-[#00E5FF]/30 text-[#00E5FF] text-xs font-medium transition-all hover:scale-105 flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3 h-3" /> Flagship Products
                  </button>
                  <button
                    onClick={() => handleSend('Tell me about CrowdIQ camera analytics')}
                    className="px-3 py-1.5 rounded-full bg-[#0F1623] hover:bg-slate-800 border border-[#00E5FF]/30 text-[#00E5FF] text-xs font-medium transition-all hover:scale-105 flex items-center gap-1.5"
                  >
                    <Eye className="w-3 h-3" /> CrowdIQ Vision
                  </button>
                  <button
                    onClick={() => handleSend('What are DevelMo global office contact details?')}
                    className="px-3 py-1.5 rounded-full bg-[#0F1623] hover:bg-slate-800 border border-[#00E5FF]/30 text-[#00E5FF] text-xs font-medium transition-all hover:scale-105 flex items-center gap-1.5"
                  >
                    <Car className="w-3 h-3" /> Global Offices
                  </button>
                </div>
              )}

              {/* Typing Indicator */}
              {isTyping && (
                <div className="flex items-center gap-2 p-2 text-slate-400 text-xs">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#00E5FF] animate-bounce" />
                    <span className="w-2 h-2 rounded-full bg-[#00E5FF] animate-bounce [animation-delay:0.15s]" />
                    <span className="w-2 h-2 rounded-full bg-[#00E5FF] animate-bounce [animation-delay:0.3s]" />
                  </div>
                  Thinking & Searching Qdrant...
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Footer */}
            <form
              onSubmit={(e: React.FormEvent<HTMLFormElement>) => {
                e.preventDefault();
                handleSend();
              }}
              className="p-3 bg-[#070A11] border-t border-[#1E293B] flex items-center gap-2"
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputQuery(e.target.value)}
                placeholder="Ask about our AI solutions..."
                className="flex-1 bg-[#0F1623] text-white placeholder-slate-500 text-sm px-4 py-3 rounded-xl border border-[#1E293B] focus:outline-none focus:border-[#00E5FF] focus:ring-1 focus:ring-[#00E5FF] transition-colors"
              />
              <button
                type="submit"
                disabled={!inputQuery.trim()}
                className="w-11 h-11 rounded-xl bg-[#00E5FF] hover:bg-[#00B8CC] text-[#070A11] flex items-center justify-center font-bold transition-all shadow-[0_0_15px_rgba(0,229,255,0.4)] active:scale-95 disabled:opacity-40"
              >
                <Send className="w-5 h-5 stroke-[2.5]" />
              </button>
            </form>
          </>
        )}
      </div>
    </>
  );
};

export default DevelMoChatWidget;
