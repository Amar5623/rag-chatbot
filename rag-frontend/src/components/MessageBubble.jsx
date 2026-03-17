import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Images proxied through Vite → FastAPI /images/ static mount
const IMAGE_BASE = ''

function Cursor() {
  return (
    <span style={{
      display: 'inline-block', width: 2, height: '1em',
      background: 'var(--accent)', marginLeft: 2,
      verticalAlign: 'text-bottom',
      animation: 'blink 1s step-end infinite',
    }} />
  )
}

function Citations({ citations }) {
  if (!citations?.length) return null
  const unique = citations.filter(
    (c, i, a) => a.findIndex(x => x.source === c.source && x.page === c.page) === i
  )
  return (
    <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 5 }}>
      {unique.map((c, i) => {
        const icon = c.chunk_type === 'image' ? '🖼' : c.chunk_type === 'table' ? '⊞' : '◈'
        const section = c.section_path || c.heading || ''
        return (
          <span key={i} style={{
            display: 'inline-flex', alignItems: 'center', gap: 5,
            background: 'var(--teal-dim)',
            border: '1px solid rgba(45,212,191,.18)',
            borderRadius: 20, padding: '3px 10px',
            fontSize: '.69rem', color: 'var(--teal)',
            fontFamily: 'var(--font-mono)', cursor: 'default',
          }}
            title={section || undefined}
          >
            <span style={{ fontSize: 10 }}>{icon}</span>
            {c.source}{c.page ? ` · p${c.page}` : ''}
            {section && (
              <span style={{ color: 'rgba(45,212,191,.5)', fontSize: '.65rem' }}>
                {section.length > 22 ? section.slice(0,22) + '…' : section}
              </span>
            )}
          </span>
        )
      })}
    </div>
  )
}

function RetrievedImages({ imageUrls }) {
  if (!imageUrls?.length) return null
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{
        fontSize: '.62rem', fontFamily: 'var(--font-mono)',
        color: 'var(--text-3)', letterSpacing: '.1em',
        textTransform: 'uppercase', marginBottom: 8,
      }}>Referenced images</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {imageUrls.map((url, i) => (
          <div key={i} style={{
            border: '1px solid var(--border)', borderRadius: 'var(--r-md)',
            overflow: 'hidden', background: 'var(--bg-2)',
          }}>
            <img
              src={`${IMAGE_BASE}${url}`}
              alt={`Figure ${i + 1}`}
              style={{ maxWidth: '100%', display: 'block', maxHeight: 320, objectFit: 'contain' }}
              onError={e => { e.target.style.display = 'none' }}
            />
            <div style={{
              padding: '4px 10px', fontSize: '.65rem',
              color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
              borderTop: '1px solid var(--border)',
            }}>{url.split('/').pop()}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function TypingDots() {
  return (
    <div style={{ display: 'flex', gap: 5, alignItems: 'center', padding: '4px 0' }}>
      {[0,1,2].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: '50%',
          background: 'var(--accent)',
          animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
          opacity: .6,
        }}/>
      ))}
    </div>
  )
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div style={{
        display: 'flex', justifyContent: 'flex-end',
        animation: 'fadeUp .22s var(--ease)', marginBottom: 12,
      }}>
        <div style={{
          maxWidth: '68%',
          background: 'linear-gradient(135deg, #1e1b3a, #17142e)',
          border: '1px solid rgba(124,106,247,.2)',
          borderRadius: '18px 18px 4px 18px',
          padding: '11px 16px',
        }}>
          <div style={{
            fontSize: '.62rem', fontFamily: 'var(--font-mono)',
            letterSpacing: '.1em', textTransform: 'uppercase',
            color: 'var(--accent-dim)', marginBottom: 5,
          }}>you</div>
          <div style={{ fontSize: '.88rem', color: 'var(--text-0)', lineHeight: 1.55 }}>
            {message.content}
          </div>
        </div>
      </div>
    )
  }

  // Assistant
  const tokens = message.usage?.total_tokens
  return (
    <div style={{
      display: 'flex', justifyContent: 'flex-start',
      animation: 'fadeUp .22s var(--ease)', marginBottom: 12,
    }}>
      {/* Avatar dot */}
      <div style={{
        width: 28, height: 28, borderRadius: 8, flexShrink: 0,
        background: 'linear-gradient(135deg, var(--accent), var(--accent-dim))',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '.75rem', color: '#fff', marginRight: 10, marginTop: 2,
      }}>✦</div>

      <div style={{
        maxWidth: 'min(78%, 700px)',
        background: 'var(--bg-2)',
        border: '1px solid var(--border-md)',
        borderRadius: '4px 18px 18px 18px',
        padding: '12px 16px',
        minWidth: 60,
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 7,
        }}>
          <div style={{
            fontSize: '.62rem', fontFamily: 'var(--font-mono)',
            letterSpacing: '.1em', textTransform: 'uppercase',
            color: 'var(--accent-text)',
          }}>docmind</div>
          {message.query_type === 'chitchat' && (
            <span style={{
              fontSize: '.6rem', fontFamily: 'var(--font-mono)',
              color: 'var(--text-3)', background: 'var(--bg-3)',
              padding: '2px 7px', borderRadius: 10,
            }}>direct</span>
          )}
        </div>

        {message.content ? (
          <div className="md" style={{ fontSize: '.87rem', color: 'var(--text-1)', lineHeight: 1.7 }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
            {message.streaming && <Cursor />}
          </div>
        ) : (
          <TypingDots />
        )}

        {!message.streaming && (<>
          <RetrievedImages imageUrls={message.image_urls} />
          <Citations citations={message.citations} />
          {tokens && (
            <div style={{
              marginTop: 8, fontSize: '.64rem', fontFamily: 'var(--font-mono)',
              color: 'var(--text-3)', textAlign: 'right',
            }}>
              {tokens.toLocaleString()} tokens
            </div>
          )}
        </>)}
      </div>
    </div>
  )
}