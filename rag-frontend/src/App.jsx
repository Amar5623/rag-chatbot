// src/App.jsx
import { useState } from 'react'
import Sidebar    from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import AuthPage   from './components/AuthPage'
import { useChat } from './hooks/useChat'
import { getToken, getUser, setToken, setUser, clearToken, clearUser } from './api'

export default function App() {
  const [token,        setTokenState]  = useState(() => getToken())
  const [currentUser,  setCurrentUser] = useState(() => getUser())
  const [kbReady,      setKbReady]     = useState(false)
  const [refreshCount, setRefreshCount] = useState(0)
  const [pinnedFile,   setPinnedFile]  = useState(null)   // ← pin state

  const { messages, streaming, statusText, send, clear } = useChat()

  const handleAuth = (data) => {
    setToken(data.access_token)
    setUser({ user_id: data.user_id, email: data.email })
    setTokenState(data.access_token)
    setCurrentUser({ user_id: data.user_id, email: data.email })
  }

  const handleLogout = () => {
    clearToken()
    clearUser()
    setTokenState(null)
    setCurrentUser(null)
    setPinnedFile(null)
    clear()
  }

  if (!token) {
    return <AuthPage onAuth={handleAuth} />
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        kbReady={kbReady}
        setKbReady={setKbReady}
        onClearChat={clear}
        refreshKey={refreshCount}
        currentUser={currentUser}
        onLogout={handleLogout}
        pinnedFile={pinnedFile}
        onPin={setPinnedFile}
        onUnpin={() => setPinnedFile(null)}
      />
      <ChatWindow
        messages={messages}
        streaming={streaming}
        statusText={statusText}
        onSend={send}
        kbReady={kbReady}
        onFilesIndexed={() => setRefreshCount(c => c + 1)}
        currentUser={currentUser}
        onLogout={handleLogout}
        pinnedFile={pinnedFile}
        onUnpin={() => setPinnedFile(null)}
      />
    </div>
  )
}