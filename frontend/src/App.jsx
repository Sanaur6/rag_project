import { useEffect, useMemo, useState } from 'react';

const sampleQuestions = [
  'What are the password requirements?',
  'How many days of leave do employees get?',
  'What is the remote work policy?',
];

const defaultRole = 'employee';

function App() {
  const [question, setQuestion] = useState('');
  const [role, setRole] = useState(defaultRole);
  const [accountRole, setAccountRole] = useState('user');
  const [activePage, setActivePage] = useState('assistant');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [feedbackData, setFeedbackData] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadFile, setUploadFile] = useState(null);

  const apiBase = useMemo(() => import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000', []);

  const fetchJson = async (url, options = {}) => {
    const response = await fetch(`${apiBase}${url}`, {
      credentials: 'include',
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload?.detail || 'Request failed.');
    }
    return payload;
  };

  const refreshHistory = async () => {
    if (!isAuthenticated) return;
    try {
      const data = await fetchJson(`/history?role=${accountRole}`);
      setHistory(data);
    } catch (err) {
      setError(err.message || 'Could not load history.');
    }
  };

  const refreshFeedback = async () => {
    if (!isAuthenticated) return;
    try {
      const data = await fetchJson(`/feedback-summary?role=${accountRole}`);
      setFeedbackData(data);
    } catch (err) {
      setError(err.message || 'Could not load feedback summary.');
    }
  };

  const refreshDocuments = async () => {
    if (!isAuthenticated || accountRole !== 'admin') return;
    try {
      const data = await fetchJson('/documents');
      setDocuments(data);
    } catch (err) {
      setError(err.message || 'Could not load documents.');
    }
  };

  useEffect(() => {
    const checkSession = async () => {
      try {
        const session = await fetchJson('/session');
        if (session?.role) {
          setAccountRole(session.role);
          setIsAuthenticated(true);
        }
      } catch {
        setIsAuthenticated(false);
      }
    };
    checkSession();
  }, [apiBase]);

  useEffect(() => {
    if (!isAuthenticated) return;
    refreshHistory();
    refreshFeedback();
    refreshDocuments();
  }, [isAuthenticated, accountRole]);

  const handleLogin = async () => {
    setLoading(true);
    setError('');

    try {
      const payload = await fetchJson('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      setAccountRole(payload.role);
      setIsAuthenticated(true);
      setUsername('');
      setPassword('');
      setActivePage('assistant');
    } catch (err) {
      setError(err.message || 'Unable to sign in.');
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await fetchJson('/logout', { method: 'POST' });
    } finally {
      setIsAuthenticated(false);
      setAccountRole('user');
      setMessages([]);
      setHistory([]);
      setFeedbackData(null);
      setDocuments([]);
      setActivePage('assistant');
    }
  };

  const askQuestion = async (nextQuestion) => {
    const trimmed = (nextQuestion || question).trim();
    if (!trimmed) return;

    const userMessage = { id: Date.now(), role: 'user', text: trimmed };
    setMessages((current) => [...current, userMessage]);
    setQuestion('');
    setLoading(true);
    setError('');

    try {
      const payload = await fetchJson('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          top_k: 4,
          user_role: role,
        }),
      });

      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          answer: payload.answer,
          action_plan: payload.action_plan || { next_steps: [] },
          sources: payload.sources || [],
        },
      ]);
      await refreshHistory();
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const saveFeedback = async (conversationId, rating, reason = null) => {
    try {
      await fetchJson('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          role: accountRole,
          rating,
          reason,
        }),
      });
      await refreshHistory();
      await refreshFeedback();
    } catch (err) {
      setError(err.message || 'Could not save feedback.');
    }
  };

  const uploadDocument = async () => {
    if (!uploadFile) return;
    try {
      const form = new FormData();
      form.append('file', uploadFile);
      await fetch(`${apiBase}/documents/upload`, {
        method: 'POST',
        credentials: 'include',
        body: form,
      });
      setUploadName('');
      setUploadFile(null);
      await refreshDocuments();
    } catch (err) {
      setError(err.message || 'Could not upload document.');
    }
  };

  const deleteDocument = async (filename) => {
    try {
      await fetch(`${apiBase}/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      await refreshDocuments();
    } catch (err) {
      setError(err.message || 'Could not delete document.');
    }
  };

  const renderAssistantPage = () => (
    <div className="page-section">
      <header className="page-header">
        <div>
          <p className="eyebrow">Knowledge Assistant</p>
          <h1>Policy questions and action guidance</h1>
        </div>
      </header>

      <div className="composer">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about a policy, benefits, HR process, or IT access requirement..."
          rows={3}
        />
        <button className="primary-button" onClick={() => askQuestion(question)} disabled={loading}>
          {loading ? 'Thinking...' : 'Ask the assistant'}
        </button>
      </div>

      <div className="sample-list-inline">
        {sampleQuestions.map((sample) => (
          <button key={sample} className="sample-button" onClick={() => setQuestion(sample)}>
            {sample}
          </button>
        ))}
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="message-list">
        {messages.map((message) => (
          <div key={message.id} className={`message-row ${message.role}`}>
            <div className="message-bubble">
              {message.role === 'user' ? (
                <p>{message.text}</p>
              ) : (
                <>
                  <p>{message.answer}</p>

                  {message.action_plan?.next_steps?.length > 0 && (
                    <div className="action-plan">
                      <strong>Recommended next steps</strong>
                      <ol>
                        {message.action_plan.next_steps.map((step, index) => (
                          <li key={`${message.id}-step-${index}`}>{step}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {message.sources?.length > 0 && (
                    <div className="sources">
                      <strong>Sources</strong>
                      {message.sources.map((source, index) => (
                        <div key={`${message.id}-source-${index}`} className="source-item">
                          <span>{source.source}</span>
                          {source.page != null && <span>Page {source.page}</span>}
                          <p>{source.text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderHistoryPage = () => (
    <div className="page-section">
      <header className="page-header">
        <div>
          <p className="eyebrow">Conversation history</p>
          <h2>Saved questions and answers</h2>
        </div>
      </header>

      {history.length === 0 ? (
        <div className="empty-card">No saved conversations yet.</div>
      ) : (
        <div className="stack-list">
          {history.map((entry) => (
            <div className="content-card" key={entry.id}>
              <div className="card-header-row">
                <strong>{entry.question}</strong>
                <span className="pill">{entry.feedback === 1 ? 'Helpful' : entry.feedback === 0 ? 'Not helpful' : 'Not rated'}</span>
              </div>
              <p>{entry.answer}</p>
              <div className="feedback-row">
                <button className="mini-button" onClick={() => saveFeedback(entry.id, 1)}>Helpful</button>
                <button className="mini-button secondary" onClick={() => saveFeedback(entry.id, 0, 'Unclear')}>Not helpful</button>
              </div>
              {entry.sources?.length > 0 && (
                <div className="tiny-sources">
                  {entry.sources.map((source, index) => (
                    <div key={`${entry.id}-source-${index}`} className="source-item compact">
                      <span>{source.source}</span>
                      {source.page != null && <span>Page {source.page}</span>}
                      <p>{source.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderFeedbackPage = () => {
    if (!feedbackData) {
      return <div className="empty-card">Loading feedback summary...</div>;
    }

    const summary = feedbackData.summary || {};
    const gapSummary = feedbackData.gap_summary || {};

    return (
      <div className="page-section">
        <header className="page-header">
          <div>
            <p className="eyebrow">Employee feedback</p>
            <h2>Quality and review insights</h2>
          </div>
        </header>

        <div className="metrics-grid">
          <div className="metric-card"><span>Total questions</span><strong>{summary.total_responses || 0}</strong></div>
          <div className="metric-card"><span>Rated answers</span><strong>{summary.rated_responses || 0}</strong></div>
          <div className="metric-card"><span>Helpful</span><strong>{summary.helpful_responses || 0}</strong></div>
          <div className="metric-card"><span>Needs review</span><strong>{summary.unhelpful_entries?.length || 0}</strong></div>
        </div>

        <div className="stack-list">
          {(gapSummary.hotspots || []).slice(0, 5).map((item, index) => (
            <div className="content-card" key={`${item.topic}-${index}`}>
              <div className="card-header-row">
                <strong>{item.topic}</strong>
                <span className="pill">{item.count} flagged</span>
              </div>
              <p><strong>Top issue:</strong> {item.most_common_reason}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderDocumentsPage = () => (
    <div className="page-section">
      <header className="page-header">
        <div>
          <p className="eyebrow">Documents</p>
          <h2>Knowledge library management</h2>
        </div>
      </header>

      <div className="upload-box">
        <input
          type="text"
          value={uploadName}
          onChange={(e) => setUploadName(e.target.value)}
          placeholder="Optional display name"
        />
        <input
          type="file"
          onChange={(e) => {
            setUploadFile(e.target.files?.[0] || null);
            setUploadName(e.target.files?.[0]?.name || '');
          }}
        />
        <button className="primary-button" onClick={uploadDocument} disabled={!uploadFile}>Upload file</button>
      </div>

      <div className="stack-list">
        {documents.length === 0 ? (
          <div className="empty-card">No documents uploaded.</div>
        ) : (
          documents.map((doc) => (
            <div className="content-card" key={doc.filename}>
              <div className="card-header-row">
                <strong>{doc.filename}</strong>
                <button className="mini-button secondary" onClick={() => deleteDocument(doc.filename)}>Delete</button>
              </div>
              <p>Type: {doc.file_type || 'TXT'}</p>
              <p>Size: {doc.size_bytes} bytes</p>
              <p>Modified: {doc.modified_at}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );

  if (!isAuthenticated) {
    return (
      <div className="login-shell">
        <div className="login-card">
          <div className="brand">
            <div className="brand-mark">AI</div>
            <div>
              <div className="brand-name">Company Knowledge</div>
              <div className="brand-subtitle">Secure session sign-in</div>
            </div>
          </div>

          <label className="field-label">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your username"
          />

          <label className="field-label">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
          />

          <button className="primary-button" onClick={handleLogin} disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>

          {error && <div className="error-box">{error}</div>}
        </div>
      </div>
    );
  }

  const pages = [
    { key: 'assistant', label: 'Assistant' },
    { key: 'history', label: 'History' },
    { key: 'feedback', label: 'Feedback' },
    ...(accountRole === 'admin' ? [{ key: 'documents', label: 'Documents' }] : []),
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <div>
            <div className="brand-name">Company Knowledge</div>
            <div className="brand-subtitle">{accountRole === 'admin' ? 'Admin workspace' : 'Employee workspace'}</div>
          </div>
        </div>

        <nav className="nav-list">
          {pages.map((page) => (
            <button
              key={page.key}
              className={`nav-button ${activePage === page.key ? 'active' : ''}`}
              onClick={() => setActivePage(page.key)}
            >
              {page.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-block">
          <label className="field-label">Ask as</label>
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="employee">Employee</option>
            <option value="manager">Manager</option>
            <option value="hr">HR</option>
            <option value="it">IT</option>
          </select>
        </div>

        <button className="secondary-button" onClick={handleLogout}>Sign out</button>
      </aside>

      <main className="main-panel">
        {activePage === 'assistant' && renderAssistantPage()}
        {activePage === 'history' && renderHistoryPage()}
        {activePage === 'feedback' && renderFeedbackPage()}
        {activePage === 'documents' && renderDocumentsPage()}
      </main>
    </div>
  );
}

export default App;
