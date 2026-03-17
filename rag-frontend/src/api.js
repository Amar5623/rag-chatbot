// src/api.js
// All calls go through the Vite proxy → FastAPI on :8000

const BASE = '/api'

// ── Health ────────────────────────────────────────────────
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

// ── Stats ─────────────────────────────────────────────────
export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

// ── Documents ─────────────────────────────────────────────
export async function fetchDocuments() {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error('Failed to fetch documents')
  return res.json()
}

// ── Ingest ────────────────────────────────────────────────
export async function ingestFiles(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Ingest failed')
  }
  return res.json()
}

// ── Wipe ──────────────────────────────────────────────────
export async function wipeCollection() {
  const res = await fetch(`${BASE}/collection`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Wipe failed')
  return res.json()
}

// ── Clear session ─────────────────────────────────────────
export async function clearSession(sessionId = 'default') {
  await fetch(`${BASE}/session/clear`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ session_id: sessionId }),
  })
}

// ── Streaming chat ────────────────────────────────────────
// Returns an async generator that yields:
//   { type: 'token',  token: string }
//   { type: 'done',   citations, query_type, usage }
//   { type: 'error',  message }
export async function* streamChat(question, sessionId = 'default') {
  const res = await fetch(`${BASE}/chat/stream`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ question, session_id: sessionId }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    yield { type: 'error', message: err.detail || 'Chat failed' }
    return
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const data = JSON.parse(raw)
        if (data.token !== undefined) {
          yield { type: 'token', token: data.token }
        } else if (data.done) {
          yield { type: 'done', ...data }   // includes image_urls, citations, usage
        } else if (data.error) {
          yield { type: 'error', message: data.error }
        }
      } catch { /* malformed line — skip */ }
    }
  }
}