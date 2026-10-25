import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Download, Share2, CheckCircle2 } from 'lucide-react';
import { InvestigationTraceViewer } from './InvestigationTraceViewer';
import type { InvestigationTraceEvent } from './InvestigationTraceViewer';

interface DossierViewerProps {
  rawReport: any;
  title?: string;
  subtitle?: string;
  modelFootprint?: string;
  metadataBadges?: Array<{ label: string; value: string | number; color?: string }>;
  investigationTrace?: InvestigationTraceEvent[];
  onExport?: () => void;
  onDispatch?: () => void;
  maxHeight?: string;
}

/**
 * Clean client-side string representation of Python lists or malformed LLM blocks
 * to ensure bulletproof presentation regardless of cache or backend history.
 */
export function cleanReportText(input: any): string {
  if (!input) return '';
  if (typeof input !== 'string') {
    if (Array.isArray(input)) {
      return input
        .map(b => (typeof b === 'object' && b !== null ? b.text || b.content || JSON.stringify(b) : String(b)))
        .join('\n\n');
    }
    return String(input);
  }

  let text = input.trim();

  // Pattern: "[{'type': 'text', 'text': '...'}]" or similar python/JSON list of dicts
  if ((text.startsWith('[') && text.endsWith(']')) || (text.startsWith('{') && text.endsWith('}'))) {
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed
          .map(b => (typeof b === 'object' && b !== null ? b.text || b.content || '' : String(b)))
          .join('\n\n');
      } else if (typeof parsed === 'object' && parsed !== null) {
        if (parsed.text) return String(parsed.text);
        if (parsed.final_report) return String(parsed.final_report);
      }
    } catch {
      // Fallback: Regex extraction for Python dict string dumps: 'text': '...' or "text": "..."
      const matches = Array.from(text.matchAll(/['"]text['"]\s*:\s*(['"])((?:(?!\1)[^\\]|\\.)*)\1/gs));
      if (matches.length > 0) {
        text = matches.map(m => m[2]).join('\n\n');
        // Unescape internal escapes
        text = text
          .replace(/\\n/g, '\n')
          .replace(/\\t/g, '\t')
          .replace(/\\"/g, '"')
          .replace(/\\'/g, "'")
          .replace(/\\\\/g, '\\');
        return text.trim();
      }
    }
  }

  // Handle literal escaped newlines "\n" if found in plain text without unescaped newlines
  if (text.includes('\\n') && !text.includes('\n')) {
    text = text.replace(/\\n/g, '\n');
  }

  return text;
}

export const DossierViewer: React.FC<DossierViewerProps> = ({
  rawReport,
  title = "FINAL RECRUITMENT INTELLIGENCE DOSSIER",
  subtitle = "COMMAND-LEVEL SCOUTING DIRECTIVE",
  modelFootprint = "GEMINI-3.5-FLASH-LITE // VERIFIED DOTA2 INTEL",
  metadataBadges = [],
  investigationTrace = [],
  onExport,
  onDispatch,
  maxHeight
}) => {
  const cleanText = useMemo(() => cleanReportText(rawReport), [rawReport]);

  if (!cleanText) {
    return (
      <div style={{
        background: '#0e1112',
        border: '1px dashed var(--border-light, #2e3538)',
        padding: '32px',
        textAlign: 'center',
        color: 'var(--text-dim, #717d82)',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '12px'
      }}>
        NO DOSSIER CONTENT RECORDED // AWAITING EVALUATION CYCLE
      </div>
    );
  }

  return (
    <div className="dossier-container" style={{
      background: '#101314',
      border: '1px solid var(--border-light, #2a3134)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
      fontFamily: 'Inter, sans-serif'
    }}>
      {/* Dossier Header Strip */}
      <div style={{
        background: '#15191b',
        borderBottom: '1px solid var(--border-dark, #1e2426)',
        padding: '16px 20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div>
          <div style={{
            fontSize: '10px',
            letterSpacing: '1.5px',
            color: 'var(--amber-gold, #f59e0b)',
            fontWeight: 800,
            textTransform: 'uppercase',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }} className="font-mono">
            <CheckCircle2 size={13} color="var(--tactical-green, #10b981)" />
            {subtitle} // {modelFootprint}
          </div>
          <h2 style={{
            fontSize: '18px',
            fontWeight: 800,
            margin: '4px 0 0 0',
            letterSpacing: '0.5px',
            color: '#ffffff'
          }} className="font-heading">
            {title}
          </h2>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {onExport && (
            <button
              className="btn-tactical"
              onClick={onExport}
              style={{ padding: '6px 12px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}
              title="Export dossier to printable format or PDF"
            >
              <Download size={13} />
              EXPORT PDF
            </button>
          )}
          {onDispatch && (
            <button
              className="btn-tactical"
              onClick={onDispatch}
              style={{
                padding: '6px 12px',
                fontSize: '11px',
                background: '#1e2427',
                borderColor: '#3a4448',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
              title="Dispatch to coaching leadership"
            >
              <Share2 size={13} />
              DISPATCH
            </button>
          )}
        </div>
      </div>

      {/* Badges Bar (if provided) */}
      {metadataBadges.length > 0 && (
        <div style={{
          display: 'flex',
          gap: '16px',
          padding: '10px 20px',
          background: '#0d1011',
          borderBottom: '1px solid var(--border-dark, #1e2426)',
          fontSize: '11px',
          fontFamily: 'JetBrains Mono, monospace',
          flexWrap: 'wrap'
        }}>
          {metadataBadges.map((badge, idx) => (
            <div key={idx} style={{ color: 'var(--text-muted, #9ca3af)' }}>
              {badge.label}: <span style={{ color: badge.color || '#fff', fontWeight: 700 }}>{badge.value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Dossier Document Body */}
      <div 
        className="dossier-content"
        style={{
          padding: '24px 28px',
          maxHeight: maxHeight || 'none',
          overflowY: maxHeight ? 'auto' : 'visible',
          fontSize: '13.5px',
          lineHeight: 1.7,
          color: '#d1d5db',
          background: '#0f1213'
        }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ node, ...props }) => (
              <h1
                style={{
                  fontFamily: 'Chakra Petch, sans-serif',
                  fontSize: '20px',
                  fontWeight: 800,
                  color: '#ffffff',
                  borderBottom: '2px solid var(--crimson-red, #dc2626)',
                  paddingBottom: '8px',
                  marginTop: '28px',
                  marginBottom: '16px',
                  letterSpacing: '0.5px',
                  textTransform: 'uppercase'
                }}
                {...props}
              />
            ),
            h2: ({ node, ...props }) => (
              <h2
                style={{
                  fontFamily: 'Chakra Petch, sans-serif',
                  fontSize: '16px',
                  fontWeight: 700,
                  color: 'var(--amber-gold, #f59e0b)',
                  borderLeft: '3px solid var(--amber-gold, #f59e0b)',
                  paddingLeft: '10px',
                  marginTop: '24px',
                  marginBottom: '12px',
                  letterSpacing: '0.5px',
                  textTransform: 'uppercase'
                }}
                {...props}
              />
            ),
            h3: ({ node, ...props }) => (
              <h3
                style={{
                  fontFamily: 'Chakra Petch, sans-serif',
                  fontSize: '14px',
                  fontWeight: 700,
                  color: '#e5e7eb',
                  marginTop: '18px',
                  marginBottom: '8px'
                }}
                {...props}
              />
            ),
            p: ({ node, ...props }) => (
              <p
                style={{
                  marginTop: '0px',
                  marginBottom: '14px',
                  color: '#d1d5db'
                }}
                {...props}
              />
            ),
            strong: ({ node, ...props }) => (
              <strong
                style={{
                  fontWeight: 700,
                  color: '#ffffff'
                }}
                {...props}
              />
            ),
            ul: ({ node, ...props }) => (
              <ul
                style={{
                  paddingLeft: '20px',
                  marginBottom: '16px',
                  marginTop: '6px'
                }}
                {...props}
              />
            ),
            ol: ({ node, ...props }) => (
              <ol
                style={{
                  paddingLeft: '22px',
                  marginBottom: '16px',
                  marginTop: '6px'
                }}
                {...props}
              />
            ),
            li: ({ node, ...props }) => (
              <li
                style={{
                  marginBottom: '6px',
                  color: '#e5e7eb'
                }}
                {...props}
              />
            ),
            blockquote: ({ node, ...props }) => (
              <blockquote
                style={{
                  borderLeft: '3px solid var(--crimson-red, #dc2626)',
                  background: 'rgba(220, 38, 38, 0.08)',
                  padding: '12px 16px',
                  margin: '16px 0',
                  color: '#f3f4f6',
                  fontStyle: 'normal'
                }}
                {...props}
              />
            ),
            table: ({ node, ...props }) => (
              <div style={{ overflowX: 'auto', margin: '20px 0' }}>
                <table
                  style={{
                    width: '100%',
                    borderCollapse: 'collapse',
                    fontSize: '12px',
                    fontFamily: 'JetBrains Mono, monospace',
                    background: '#0a0c0d',
                    border: '1px solid #23282b'
                  }}
                  {...props}
                />
              </div>
            ),
            th: ({ node, ...props }) => (
              <th
                style={{
                  background: '#191e21',
                  color: 'var(--amber-gold, #f59e0b)',
                  padding: '8px 12px',
                  textAlign: 'left',
                  borderBottom: '2px solid #2e3538',
                  fontWeight: 700,
                  letterSpacing: '0.5px'
                }}
                {...props}
              />
            ),
            td: ({ node, ...props }) => (
              <td
                style={{
                  padding: '8px 12px',
                  borderBottom: '1px solid #1c2123',
                  color: '#e5e7eb'
                }}
                {...props}
              />
            ),
            code: ({ node, ...props }) => (
              <code
                style={{
                  background: '#181d1f',
                  color: 'var(--tactical-green, #10b981)',
                  padding: '2px 6px',
                  borderRadius: '2px',
                  fontSize: '12px',
                  fontFamily: 'JetBrains Mono, monospace'
                }}
                {...props}
              />
            ),
            hr: () => (
              <hr
                style={{
                  border: 'none',
                  borderTop: '1px solid var(--border-dark, #1e2426)',
                  margin: '24px 0'
                }}
              />
            )
          }}
        >
          {cleanText}
        </ReactMarkdown>
      </div>

      {/* Optional Investigation Trace Section integrated directly into the Dossier */}
      {investigationTrace && investigationTrace.length > 0 && (
        <div style={{ padding: '0 28px 20px 28px' }}>
          <InvestigationTraceViewer 
            traceEvents={investigationTrace} 
            title="INVESTIGATION TRACE // VERIFIED TOOL DECISIONS & EVIDENCE"
          />
        </div>
      )}

      {/* Dossier Footer / Confidentiality notice */}
      <div style={{
        background: '#0c0e0f',
        borderTop: '1px solid var(--border-dark, #1e2426)',
        padding: '10px 20px',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: 'var(--text-dim, #5c666a)',
        fontFamily: 'JetBrains Mono, monospace'
      }}>
        <span>RESTRICTED SCOUTING ASSET // ROASTERLENS v1.1</span>
        <span>GROUND TRUTH: OPENDOTA API LIVE MATCH TELEMETRY</span>
      </div>
    </div>
  );
};
