import { useState } from 'react'
import Sidebar    from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import { useChat } from './hooks/useChat'

export default function App() {
  const [kbReady,      setKbReady]      = useState(false)
  // Incrementing this triggers Sidebar to re-fetch stats
  const [refreshCount, setRefreshCount] = useState(0)
  const { messages, streaming, statusText, send, clear } = useChat('default')

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        kbReady={kbReady}
        setKbReady={setKbReady}
        onClearChat={clear}
        refreshKey={refreshCount}
      />
      <ChatWindow
        messages={messages}
        streaming={streaming}
        statusText={statusText}
        onSend={send}
        kbReady={kbReady}
        onFilesIndexed={() => setRefreshCount(c => c + 1)}
      />
    </div>
  )
}