// src/components/AuthPage.jsx
// Login / Signup page — single component with mode toggle.
// Matches the existing dark theme (CSS variables from index.css).
// On success calls onAuth({ token, user_id, email }).

import { useState } from 'react'
import { login, signup } from '../api'

export default function AuthPage({ onAuth }) {
  const [mode,     setMode]     = useState('login')   // 'login' | 'signup'
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [confirm,  setConfirm]  = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  const isLogin  = mode === 'login'
  const isSignup = mode === 'signup'

  const handleSubmit = async () => {
    setError('')

    if (!email.trim() || !password) {
      setError('Email and password are required.')
      return
    }
    if (isSignup && password !== confirm) {
      setError('Passwords do not match.')
      return
    }
    if (isSignup && password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }

    setLoading(true)
    try {
      const data = isLogin
        ? await login(email.trim(), password)
        : await signup(email.trim(), password)
      onAuth(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = e => {
    if (e.key === 'Enter') handleSubmit()
  }

  return (
    <div style={{
      width: '100vw', height: '100vh',
      background: 'var(--bg-0)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        width: 380,
        background: 'var(--bg-1)',
        border: '1px solid var(--border-md)',
        borderRadius: 'var(--r-xl)',
        padding: '36px 32px',
        animation: 'fadeUp .35s var(--ease)',
      }}>

        {/* Logo + title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 11, marginBottom: 28 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9, flexShrink: 0,
            background: 'linear-gradient(135deg, var(--accent) 0%, #5b4dd4 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: '#fff',
          }}>✦</div>
          <div>
            <div style={{
              fontFamily: 'var(--font-display)', fontWeight: 800,
              fontSize: '1.1rem', color: 'var(--text-0)',
            }}>DocMind</div>
            <div style={{
              fontSize: '.6rem', color: 'var(--text-3)',
              letterSpacing: '.1em', textTransform: 'uppercase',
            }}>RAG Intelligence</div>
          </div>
        </div>

        {/* Mode tabs */}
        <div style={{
          display: 'flex', gap: 4, marginBottom: 24,
          background: 'var(--bg-3)', borderRadius: 'var(--r-md)',
          padding: 4,
        }}>
          {['login', 'signup'].map(m => (
            <button
              key={m}
              onClick={() => { setMode(m); setError('') }}
              style={{
                flex: 1, padding: '7px 0',
                borderRadius: 'var(--r-sm)',
                border: 'none', cursor: 'pointer',
                fontFamily: 'var(--font-display)',
                fontWeight: 700, fontSize: '.78rem',
                letterSpacing: '.04em',
                background: mode === m ? 'var(--bg-1)' : 'transparent',
                color     : mode === m ? 'var(--text-0)' : 'var(--text-3)',
                boxShadow : mode === m ? '0 1px 4px rgba(0,0,0,.35)' : 'none',
                transition: 'all .15s',
              }}
            >
              {m === 'login' ? 'Sign in' : 'Create account'}
            </button>
          ))}
        </div>

        {/* Fields */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Field
            label="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            onKeyDown={handleKey}
            placeholder="you@example.com"
          />
          <Field
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={handleKey}
            placeholder={isSignup ? 'Minimum 6 characters' : '••••••••'}
          />
          {isSignup && (
            <Field
              label="Confirm password"
              type="password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Repeat password"
            />
          )}
        </div>

        {/* Error */}
        {error && (
          <div style={{
            marginTop: 14,
            padding: '9px 13px',
            borderRadius: 'var(--r-md)',
            background: 'rgba(239,68,68,.1)',
            border: '1px solid rgba(239,68,68,.25)',
            fontSize: '.78rem', color: 'rgba(239,100,100,.9)',
          }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          style={{
            width: '100%', marginTop: 20,
            padding: '11px 0',
            borderRadius: 'var(--r-md)',
            border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
            background: loading
              ? 'var(--bg-3)'
              : 'linear-gradient(135deg, var(--accent), var(--accent-dim))',
            color: loading ? 'var(--text-3)' : '#fff',
            fontFamily: 'var(--font-display)',
            fontWeight: 700, fontSize: '.88rem',
            letterSpacing: '.04em',
            transition: 'all .15s',
            opacity: loading ? .6 : 1,
          }}
        >
          {loading
            ? (isLogin ? 'Signing in…' : 'Creating account…')
            : (isLogin ? 'Sign in' : 'Create account')}
        </button>

      </div>
    </div>
  )
}

/* ── Labelled input ── */
function Field({ label, type, value, onChange, onKeyDown, placeholder }) {
  return (
    <div>
      <div style={{
        fontSize: '.68rem', fontFamily: 'var(--font-mono)',
        letterSpacing: '.1em', textTransform: 'uppercase',
        color: 'var(--text-3)', marginBottom: 6,
      }}>{label}</div>
      <input
        type={type}
        value={value}
        onChange={onChange}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        style={{
          width: '100%', padding: '9px 13px',
          background: 'var(--bg-3)',
          border: '1px solid var(--border-md)',
          borderRadius: 'var(--r-md)',
          color: 'var(--text-0)',
          fontFamily: 'var(--font-body)', fontSize: '.88rem',
          outline: 'none',
          transition: 'border-color .15s',
        }}
        onFocus={e  => e.target.style.borderColor = 'var(--accent-dim)'}
        onBlur={e   => e.target.style.borderColor = 'var(--border-md)'}
      />
    </div>
  )
}