import React, { useState } from 'react';
import { 
  ChevronDown, ChevronRight, Terminal, Database, Cpu, 
  Search, Activity, Split
} from 'lucide-react';

export interface ToolCallExplanation {
  tool_name: string;
  step_number: number;
  step_title: string;
  reason: string;
  input_params: Record<string, any>;
  result_summary: string;
  impact: string;
  status: 'completed' | 'failed' | 'skipped' | 'running';
  latency_ms?: number;
  error?: string | null;
  data_source_type: 'opendota_api' | 'python_metric' | 'web_research' | 'llm_reasoning';
}

export interface InvestigationTraceEvent {
  investigation_id: string;
  step_id: string;
  step_order: number;
  title: string;
  node_name: string;
  decision_summary: string;
  evidence_gap_or_objective: string;
  status: 'completed' | 'failed' | 'skipped' | 'waiting_for_user';
  tool_calls: ToolCallExplanation[];
  conditional_decision?: string | null;
  duration_ms?: number;
  timestamp: string;
  error_details?: string | null;
}

interface InvestigationTraceViewerProps {
  traceEvents: InvestigationTraceEvent[];
  title?: string;
  isLive?: boolean;
}

export const InvestigationTraceViewer: React.FC<InvestigationTraceViewerProps> = ({
  traceEvents,
  title = "INVESTIGATION DECISION TRACE & TOOL-CALL EXPLANATIONS",
  isLive = false
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({
    "collect_opendota_evidence": true,
    "synthesize_evidence": true
  });

  const toggleStep = (stepId: string) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepId]: !prev[stepId]
    }));
  };

  const getSourceBadge = (type: ToolCallExplanation['data_source_type']) => {
    switch (type) {
      case 'opendota_api':
        return { label: 'OPENDOTA API', color: '#38bdf8', icon: <Database size={11} /> };
      case 'python_metric':
        return { label: 'PYTHON CALC', color: 'var(--amber-gold)', icon: <Cpu size={11} /> };
      case 'web_research':
        return { label: 'WEB RESEARCH', color: '#c084fc', icon: <Search size={11} /> };
      case 'llm_reasoning':
      default:
        return { label: 'LANGGRAPH / LLM', color: 'var(--tactical-green)', icon: <Terminal size={11} /> };
    }
  };

  if (!traceEvents || traceEvents.length === 0) {
    return (
      <div style={{
        background: '#0d1011',
        border: '1px dashed var(--border-light)',
        padding: '24px',
        textAlign: 'center',
        color: 'var(--text-dim)',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px'
      }}>
        [NO EXECUTION TRACE RECORDED // DISPATCH PIPELINE TO STREAM TIMELINE]
      </div>
    );
  }

  return (
    <div style={{
      background: '#0c0f10',
      border: '1px solid var(--border-light)',
      boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      fontFamily: 'Inter, sans-serif',
      marginTop: '20px',
      marginBottom: '24px'
    }}>
      {/* Header bar */}
      <div style={{
        background: '#14181a',
        borderBottom: '1px solid var(--border-dark)',
        padding: '14px 18px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '8px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={14} color="var(--tactical-green)" />
          <span style={{ fontSize: '11px', color: 'var(--amber-gold)', fontWeight: 800, letterSpacing: '1px' }} className="font-mono">
            {title}
          </span>
          {isLive && (
            <span style={{ fontSize: '9px', background: 'var(--crimson-red-dim)', color: 'var(--crimson-red)', padding: '2px 6px', fontWeight: 700 }} className="font-mono">
              LIVE STREAMING
            </span>
          )}
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-dim)' }} className="font-mono">
          TRACE NODES: <span style={{ color: '#fff', fontWeight: 700 }}>{traceEvents.length}</span>
        </div>
      </div>

      {/* Ground-truth source guide pill */}
      <div style={{
        background: '#080a0b',
        borderBottom: '1px solid var(--border-dark)',
        padding: '8px 18px',
        display: 'flex',
        gap: '16px',
        fontSize: '10px',
        color: 'var(--text-dim)',
        fontFamily: 'JetBrains Mono',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <span style={{ color: 'var(--text-muted)' }}>GROUND TRUTH TAXONOMY:</span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#38bdf8' }}>
          <Database size={10} /> OPENDOTA API (Empirical)
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--amber-gold)' }}>
          <Cpu size={10} /> PYTHON METRIC (Calculated)
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--tactical-green)' }}>
          <Terminal size={10} /> LANGGRAPH / LLM (Strategic)
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#c084fc' }}>
          <Search size={10} /> WEB INTEL (External)
        </span>
      </div>

      {/* Timeline items */}
      <div style={{ padding: '16px' }}>
        {traceEvents.map((evt, idx) => {
          const isExpanded = !!expandedSteps[evt.step_id];
          const hasTools = evt.tool_calls && evt.tool_calls.length > 0;

          return (
            <div 
              key={idx} 
              style={{
                background: '#101315',
                border: '1px solid var(--border-dark)',
                marginBottom: '12px',
                transition: 'border-color 0.2s ease'
              }}
            >
              {/* Step Title Header (Clickable Accordion) */}
              <div 
                onClick={() => toggleStep(evt.step_id)}
                style={{
                  padding: '12px 16px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                  userSelect: 'none',
                  background: isExpanded ? '#151a1c' : '#101315'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {isExpanded ? <ChevronDown size={14} color="var(--amber-gold)" /> : <ChevronRight size={14} color="var(--text-dim)" />}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }} className="font-heading">
                        {evt.title || `Step ${evt.step_order} — ${evt.node_name}`}
                      </span>
                      <span style={{ 
                        fontSize: '9px', 
                        padding: '1px 6px', 
                        borderRadius: '2px',
                        background: evt.status === 'completed' ? 'var(--tactical-green-dim)' : 'var(--crimson-red-dim)',
                        color: evt.status === 'completed' ? 'var(--tactical-green)' : 'var(--crimson-red)',
                        fontWeight: 700 
                      }} className="font-mono">
                        {evt.status.toUpperCase()}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }} className="font-mono">
                      {evt.decision_summary}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }} className="font-mono">
                  {hasTools && (
                    <span style={{ fontSize: '10px', color: 'var(--amber-gold)', background: '#1c2225', padding: '2px 8px' }}>
                      {evt.tool_calls.length} {evt.tool_calls.length === 1 ? 'TOOL CALL' : 'TOOL CALLS'}
                    </span>
                  )}
                  <span style={{ fontSize: '10px', color: 'var(--text-dim)' }}>
                    {evt.timestamp ? evt.timestamp.slice(11, 19) : ''}
                  </span>
                </div>
              </div>

              {/* Step Expanded Content */}
              {isExpanded && (
                <div style={{ padding: '16px', borderTop: '1px solid var(--border-dark)', background: '#0b0e0f' }}>
                  {/* Task Objective / Evidence Gap */}
                  {evt.evidence_gap_or_objective && (
                    <div style={{ marginBottom: '14px', background: '#111517', padding: '10px 14px', borderLeft: '3px solid var(--amber-gold)' }}>
                      <div style={{ fontSize: '9px', color: 'var(--amber-gold)', fontWeight: 800 }} className="font-mono">
                        EVIDENCE GAP / TASK OBJECTIVE:
                      </div>
                      <div style={{ fontSize: '12px', color: '#d1d5db', marginTop: '3px', lineHeight: 1.5 }}>
                        {evt.evidence_gap_or_objective}
                      </div>
                    </div>
                  )}

                  {/* Conditional Decision (Why another tool was called / branched) */}
                  {evt.conditional_decision && (
                    <div style={{ marginBottom: '14px', background: '#111618', padding: '10px 14px', borderLeft: '3px solid #38bdf8' }}>
                      <div style={{ fontSize: '9px', color: '#38bdf8', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '4px' }} className="font-mono">
                        <Split size={11} />
                        CONDITIONAL DECISION CONTEXT:
                      </div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1', marginTop: '3px', lineHeight: 1.5 }}>
                        {evt.conditional_decision}
                      </div>
                    </div>
                  )}

                  {/* Tool Call Cards */}
                  {hasTools && (
                    <div style={{ marginTop: '12px' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 800, marginBottom: '8px' }} className="font-mono">
                        DETAILED TOOL INVOCATIONS & RESULTS ({evt.tool_calls.length})
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {evt.tool_calls.map((tool, tIdx) => {
                          const badge = getSourceBadge(tool.data_source_type);

                          return (
                            <div 
                              key={tIdx} 
                              style={{ 
                                background: '#131719', 
                                border: '1px solid var(--border-dark)', 
                                padding: '12px',
                                borderLeft: `3px solid ${badge.color}`
                              }}
                            >
                              {/* Tool Name Bar */}
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <span style={{ fontSize: '12px', fontWeight: 800, color: '#fff' }} className="font-mono">
                                    {tool.step_title || tool.tool_name}
                                  </span>
                                  <span style={{ 
                                    fontSize: '9px', 
                                    color: badge.color, 
                                    background: '#192024', 
                                    padding: '1px 6px', 
                                    display: 'inline-flex', 
                                    alignItems: 'center', 
                                    gap: '3px',
                                    fontWeight: 700
                                  }} className="font-mono">
                                    {badge.icon}
                                    {badge.label}
                                  </span>
                                </div>
                                <span style={{ fontSize: '9px', color: tool.status === 'completed' ? 'var(--tactical-green)' : 'var(--crimson-red)', fontWeight: 700 }} className="font-mono">
                                  {tool.status.toUpperCase()}
                                </span>
                              </div>

                              {/* Tool Call Breakdown (Step, Reason, Input, Result, Impact) */}
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '10px', fontSize: '11px', fontFamily: 'Inter, sans-serif' }}>
                                {/* REASON */}
                                <div style={{ background: '#0e1113', padding: '8px', border: '1px solid #1a2023' }}>
                                  <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontWeight: 800 }} className="font-mono">REASON / EVIDENCE GAP</div>
                                  <div style={{ color: '#d1d5db', marginTop: '3px', lineHeight: 1.4 }}>{tool.reason}</div>
                                </div>

                                {/* INPUT */}
                                <div style={{ background: '#0e1113', padding: '8px', border: '1px solid #1a2023' }}>
                                  <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontWeight: 800 }} className="font-mono">INPUT PARAMETERS</div>
                                  <div style={{ color: '#38bdf8', marginTop: '3px', fontFamily: 'JetBrains Mono', fontSize: '10.5px' }}>
                                    {Object.keys(tool.input_params || {}).length > 0 
                                      ? JSON.stringify(tool.input_params) 
                                      : 'None (default query)'}
                                  </div>
                                </div>

                                {/* RESULT */}
                                <div style={{ background: '#0e1113', padding: '8px', border: '1px solid #1a2023' }}>
                                  <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontWeight: 800 }} className="font-mono">RESULT SUMMARY</div>
                                  <div style={{ color: '#e5e7eb', marginTop: '3px', lineHeight: 1.4 }}>{tool.result_summary}</div>
                                </div>

                                {/* IMPACT */}
                                <div style={{ background: '#0e1113', padding: '8px', border: '1px solid #1a2023' }}>
                                  <div style={{ fontSize: '9px', color: 'var(--text-dim)', fontWeight: 800 }} className="font-mono">EVALUATION CONTRIBUTION</div>
                                  <div style={{ color: 'var(--amber-gold)', marginTop: '3px', lineHeight: 1.4 }}>{tool.impact}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
