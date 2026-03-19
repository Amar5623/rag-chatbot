// src/hooks/useChat.js
//
// CHANGES:
//   - sessionId param removed — backend now derives session from JWT user_id
//   - streamChat() called without sessionId argument
//   - Everything else identical

import { useState, useCallback, useRef } from 'react'
import { streamChat, clearSession } from '../api'

export function useChat() {
  const [messages,   setMessages]   = useState([])
  const [streaming,  setStreaming]  = useState(false)
  const [statusText, setStatusText] = useState('')
  const abortRef = useRef(false)

  const send = useCallback(async (question) => {
    if (streaming) return
    abortRef.current = false

    const userMsg = { id: Date.now(), role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])

    const assistantId = Date.now() + 1
    setMessages(prev => [...prev, {
      id: assistantId, role: 'assistant', content: '', streaming: true,
      citations: [], image_urls: [], query_type: 'document', usage: {},
    }])

    setStreaming(true)
    setStatusText('Searching documents…')

    try {
      let firstToken = false
      for await (const event of streamChat(question)) {
        if (abortRef.current) break

        if (event.type === 'token') {
          if (!firstToken) { firstToken = true; setStatusText('') }
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: m.content + event.token }
              : m
          ))
        } else if (event.type === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, streaming: false,
                  citations  : event.citations   || [],
                  image_urls : event.image_urls  || [],
                  query_type : event.query_type  || 'document',
                  usage      : event.usage       || {} }
              : m
          ))
        } else if (event.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `⚠️ ${event.message}`, streaming: false, isError: true }
              : m
          ))
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: `⚠️ ${err.message}`, streaming: false, isError: true }
          : m
      ))
    } finally {
      setStreaming(false)
      setStatusText('')
    }
  }, [streaming])

  const clear = useCallback(async () => {
    await clearSession()
    setMessages([])
  }, [])

  return { messages, streaming, statusText, send, clear }
}