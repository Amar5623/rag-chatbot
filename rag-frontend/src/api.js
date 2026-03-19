// src/api.js
// All calls go through the Vite proxy → FastAPI on :8000
// JWT token is read from localStorage and attached to every protected request.

const BASE = '/api'

// ── Token helpers ─────────────────────────────────────────────
export const getToken  = ()          => localStorage.getItem('rag_token')
export const setToken  = (t)         => localStorage.setItem('rag_token', t)
export const clearToken = ()         => localStorage.removeItem('rag_token')
export const getUser   = ()          => {
  try { return JSON.parse(localStorage.getItem('rag_user') || 'null') }
  catch { return null }
}
export const setUser   = (u)         => localStorage.setItem('rag_user', JSON.stringify(u))
export const clearUser = ()          => localStorage.removeItem('rag_user')

function authHeaders(extra = {}) {
  const token = getToken()
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  }
}

// ── Auth ──────────────────────────────────────────────────────
export async function signup(email, password) {
  const res = await fetch(`${BASE}/auth/signup`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ email, password }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Signup failed')
  return data   // { access_token, user_id, email }
}

export async function login(email, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ email, password }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Login failed')
  return data   // { access_token, user_id, email }
}

// ── Health ────────────────────────────────────────────────────
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

// ── Stats (public — sidebar polls this) ───────────────────────
export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

// ── Documents ─────────────────────────────────────────────────
export async function fetchDocuments() {
  const res = await fetch(`${BASE}/documents`, {
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error('Failed to fetch documents')
  return res.json()
}

// ── Ingest ────────────────────────────────────────────────────
export async function ingestFiles(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch(`${BASE}/ingest`, {
    method : 'POST',
    headers: authHeaders(),   // no Content-Type — browser sets multipart boundary
    body   : form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Ingest failed')
  }
  return res.json()
}

// ── Delete file ───────────────────────────────────────────────
export async function deleteFile(filename) {
  const res = await fetch(`${BASE}/ingest/${encodeURIComponent(filename)}`, {
    method : 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Delete failed')
  }
  return res.json()
}

// ── Wipe ──────────────────────────────────────────────────────
export async function wipeCollection() {
  const res = await fetch(`${BASE}/collection`, {
    method : 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error('Wipe failed')
  return res.json()
}

// ── Clear session ─────────────────────────────────────────────
export async function clearSession() {
  await fetch(`${BASE}/session/clear`, {
    method : 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body   : JSON.stringify({ session_id: 'ignored' }),  // backend uses JWT
  })
}

// ── Streaming chat ────────────────────────────────────────────
// Yields: { type: 'token', token }  |  { type: 'done', ... }  |  { type: 'error', message }
export async function* streamChat(question) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method : 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body   : JSON.stringify({ question, session_id: 'ignored' }),
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
    buffer = lines.pop()

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const data = JSON.parse(raw)
        if (data.token !== undefined) {
          yield { type: 'token', token: data.token }
        } else if (data.done) {
          yield { type: 'done', ...data }
        } else if (data.error) {
          yield { type: 'error', message: data.error }
        }
      } catch { /* malformed line — skip */ }
    }
  }
}

// ── Pin / Unpin source ────────────────────────────────────────
export async function pinFile(filename) {
  const res = await fetch(`${BASE}/session/pin`, {
    method : 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body   : JSON.stringify({ filename }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Pin failed')
  }
  return res.json()
}

export async function unpinFile() {
  const res = await fetch(`${BASE}/session/pin`, {
    method : 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error('Unpin failed')
  return res.json()
}