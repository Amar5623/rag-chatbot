import { useState, useEffect } from 'react'
import { fetchStats, ingestFiles, wipeCollection } from '../api'

export default function Sidebar({ onClearChat, kbReady, setKbReady, refreshKey }) {
  const [collapsed, setCollapsed] = useState(false)
  const [stats,     setStats]     = useState(null)
  const [drag,      setDrag]      = useState(false)
  const [pending,   setPending]   = useState([])
  const [busy,      setBusy]      = useState(false)
  const [busyMsg,   setBusyMsg]   = useState('')

  const refresh = async () => {
    try {
      const s = await fetchStats()
      setStats(s)
      setKbReady(s.total_vectors > 0)
    } catch { /* backend not ready yet */ }
  }

  // Fetch on mount + whenever refreshKey increments (after ChatWindow upload)
  useEffect(() => { refresh() }, [refreshKey])

  /* ── File handling ── */

  // Using a unique id for the label→input pattern — always reliable cross-browser
  const INPUT_ID = 'sidebar-file-input'

  const addFiles = files => {
    const valid = [...files].filter(f =>
      ['.pdf', '.csv', '.xlsx', '.txt'].some(x => f.name.toLowerCase().endsWith(x))
    )
    if (valid.length) setPending(p => [...p, ...valid])
  }

  const handleDrop = e => {
    e.preventDefault(); setDrag(false)
    addFiles(e.dataTransfer.files)
  }

  // label→input onChange — fires reliably because browser owns the click
  const handleChange = e => {
    addFiles(e.target.files)
    // Reset so the same file can be selected again
    e.target.value = ''
  }

  const removeFile = i => setPending(p => p.filter((_, j) => j !== i))

  const doIngest = async () => {
    if (!pending.length) return
    setBusy(true); setBusyMsg(`Indexing ${pending.length} file(s)…`)
    try {
      await ingestFiles(pending); setPending([]); await refresh()
    } catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const doWipe = async () => {
    if (!confirm('Wipe the entire knowledge base? This cannot be undone.')) return
    setBusy(true); setBusyMsg('Wiping…')
    try { await wipeCollection(); await refresh(); onClearChat() }
    catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const extIcon = n => {
    const l = n.toLowerCase()
    if (l.endsWith('.pdf'))  return '📄'
    if (l.endsWith('.csv'))  return '📊'
    if (l.endsWith('.xlsx')) return '📗'
    return '📝'
  }

  /* ── Styles ── */
  const W = collapsed ? 52 : 268

  return (
    <aside style={{
      width: W, minWidth: W,
      background: 'var(--bg-1)',
      borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      height: '100vh', overflow: 'hidden',
      transition: 'width .25s cubic-bezier(.16,1,.3,1)',
      flexShrink: 0,
    }}>

      {/* ── Header ── */}
      <div style={{
        height: 56, padding: '0 14px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center',
        justifyContent: collapsed ? 'center' : 'space-between',
        flexShrink: 0,
      }}>
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, overflow: 'hidden' }}>
            <Logo />
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '.95rem', color: 'var(--text-0)', whiteSpace: 'nowrap' }}>
                DocMind
              </div>
              <div style={{ fontSize: '.58rem', color: 'var(--text-3)', letterSpacing: '.1em', textTransform: 'uppercase' }}>
                RAG Intelligence
              </div>
            </div>
          </div>
        )}
        {collapsed && <Logo />}

        {/* Collapse toggle */}
        <button onClick={() => setCollapsed(c => !c)} style={{
          width: 26, height: 26, borderRadius: 6,
          border: '1px solid var(--border)',
          background: 'transparent', cursor: 'pointer',
          color: 'var(--text-2)', display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          fontSize: '.75rem', flexShrink: 0,
          transition: 'all .15s',
        }}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? '›' : '‹'}
        </button>
      </div>

      {/* ── Body (hidden when collapsed) ── */}
      {!collapsed && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px' }}>

          {/* Status pill */}
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 16,
            background: kbReady ? 'rgba(45,212,191,.08)' : 'rgba(255,255,255,.03)',
            border: `1px solid ${kbReady ? 'rgba(45,212,191,.2)' : 'var(--border)'}`,
            borderRadius: 20, padding: '4px 11px',
            fontSize: '.68rem', color: kbReady ? 'var(--teal)' : 'var(--text-3)',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
              background: kbReady ? 'var(--teal)' : 'var(--text-3)',
              animation: kbReady ? 'pulse 2.5s ease infinite' : 'none',
            }}/>
            {kbReady ? 'KB ready' : 'No documents'}
          </div>

          <SLabel>Upload documents</SLabel>

          {/* Drop zone — label wraps the hidden input for reliable browser file dialog */}
          <label
            htmlFor={INPUT_ID}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={handleDrop}
            style={{
              display: 'block',
              border: `1.5px dashed ${drag ? 'var(--accent)' : 'var(--border-md)'}`,
              borderRadius: 'var(--r-lg)', padding: '18px 12px',
              textAlign: 'center', cursor: 'pointer',
              background: drag ? 'var(--accent-glow)' : 'var(--bg-0)',
              transition: 'all .2s', marginBottom: 8,
            }}
          >
            <div style={{ fontSize: '1.2rem', marginBottom: 5, opacity: .7 }}>⊕</div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-2)', lineHeight: 1.5 }}>
              Drop files or <span style={{ color: 'var(--accent-text)' }}>browse</span>
            </div>
            <div style={{ fontSize: '.64rem', color: 'var(--text-3)', marginTop: 3, fontFamily: 'var(--font-mono)' }}>
              pdf · csv · xlsx · txt
            </div>
          </label>

          {/* Hidden file input — triggered by label above */}
          <input
            id={INPUT_ID}
            type="file"
            multiple
            accept=".pdf,.csv,.xlsx,.txt"
            onChange={handleChange}
            style={{ display: 'none' }}
          />

          {/* Pending file list */}
          {pending.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginBottom: 8 }}>
              {pending.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 7,
                  background: 'var(--bg-2)', border: '1px solid var(--border)',
                  borderRadius: 'var(--r-sm)', padding: '5px 9px',
                  fontSize: '.71rem', color: 'var(--text-1)',
                }}>
                  <span style={{ fontSize: 11, flexShrink: 0 }}>{extIcon(f.name)}</span>
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {f.name}
                  </span>
                  <span
                    onClick={e => { e.preventDefault(); removeFile(i) }}
                    style={{ cursor: 'pointer', color: 'var(--text-3)', lineHeight: 1, flexShrink: 0 }}
                  >✕</span>
                </div>
              ))}
            </div>
          )}

          <SBtn primary onClick={doIngest} disabled={!pending.length || busy}>
            {busy ? busyMsg : pending.length
              ? `Index ${pending.length} file${pending.length > 1 ? 's' : ''}`
              : 'Select files above'}
          </SBtn>

          {/* Chat actions */}
          {kbReady && (<>
            <SLabel>Chat</SLabel>
            <SBtn onClick={onClearChat} style={{ marginBottom: 6 }}>Clear conversation</SBtn>
            <SBtn danger onClick={doWipe}>Wipe knowledge base</SBtn>
          </>)}

          {/* Stats */}
          {stats && (<>
            <SLabel>Stats</SLabel>
            {[
              ['Vectors',  stats.total_vectors],
              ['BM25',     stats.bm25_docs],
              ['Parents',  stats.parent_count],
              ['Model',    stats.llm_model?.split('-').slice(0,3).join('-') + '…'],
              ['Embedder', stats.embedding_model?.split('/').pop()],
            ].map(([l, v]) => (
              <div key={l} style={{
                display: 'flex', justifyContent: 'space-between',
                padding: '5px 0', borderBottom: '1px solid var(--border)',
                fontSize: '.72rem',
              }}>
                <span style={{ color: 'var(--text-2)' }}>{l}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '.69rem', color: 'var(--accent-text)' }}>{v}</span>
              </div>
            ))}

            {stats.indexed_files?.length > 0 && (<>
              <SLabel>Indexed files</SLabel>
              {stats.indexed_files.map(f => (
                <div key={f} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 0', borderBottom: '1px solid var(--border)',
                  fontSize: '.71rem', color: 'var(--text-2)',
                }}>
                  <span style={{ fontSize: 11 }}>{extIcon(f)}</span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f}</span>
                </div>
              ))}
            </>)}
          </>)}
        </div>
      )}

      {collapsed && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 14, gap: 12 }}>
          <label htmlFor={INPUT_ID} title="Upload files" style={{
            width: 32, height: 32, borderRadius: 8, cursor: 'pointer',
            background: 'var(--bg-3)', border: '1px solid var(--border-md)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: 'var(--text-1)',
          }}>⊕</label>
          {/* Input already rendered above via INPUT_ID — label just links to it */}
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: kbReady ? 'var(--teal)' : 'var(--text-3)',
            animation: kbReady ? 'pulse 2.5s ease infinite' : 'none',
          }} title={kbReady ? 'KB ready' : 'No documents'}/>
        </div>
      )}
    </aside>
  )
}

/* ── Helpers ── */

function Logo() {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: 7, flexShrink: 0,
      background: 'linear-gradient(135deg, var(--accent) 0%, #5b4dd4 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '.85rem', color: '#fff',
    }}>✦</div>
  )
}

function SLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: '.59rem', fontWeight: 500,
      letterSpacing: '.12em', textTransform: 'uppercase', color: 'var(--text-3)',
      margin: '18px 0 8px', paddingBottom: 5, borderBottom: '1px solid var(--border)',
    }}>{children}</div>
  )
}

function SBtn({ children, onClick, disabled, primary, danger, style }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      width: '100%', padding: '8px 12px', borderRadius: 'var(--r-md)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontFamily: 'var(--font-display)', fontWeight: 700,
      fontSize: '.72rem', letterSpacing: '.03em',
      transition: 'all .15s', marginBottom: 6, opacity: disabled ? .5 : 1,
      border: primary ? 'none'
        : danger ? '1px solid rgba(239,68,68,.25)'
        : '1px solid var(--border-md)',
      background: primary
        ? 'linear-gradient(135deg, var(--accent), var(--accent-dim))'
        : 'transparent',
      color: primary ? '#fff' : danger ? 'rgba(239,100,100,.85)' : 'var(--text-1)',
      ...style,
    }}>{children}</button>
  )
}