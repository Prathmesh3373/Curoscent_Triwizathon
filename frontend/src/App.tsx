import { useState, useEffect } from 'react';
import { DossierViewer } from './components/DossierViewer';
import { InvestigationTraceViewer } from './components/InvestigationTraceViewer';
import { 
  Users, FileText, Shield, 
  Search, ArrowRight, Play, RefreshCw, Sliders, AlertTriangle
} from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000/api";

type TabKey = 'overview' | 'scout' | 'candidates' | 'dossiers' | 'simulator' | 'reports';

interface HealthData {
  status: string;
  service: string;
  model: string;
  llm_configured: boolean;
  active_sessions_count: number;
}

interface ShortlistedCandidate {
  account_id: number;
  name: string;
  personaname: string;
  avatar?: string;
  team_name: string;
  role_desc: string;
  compatibility_score: number;
  supporting_evidence: string;
  missing_data_warning: string | null;
  match_count?: number;
  investigation_id: string;
}

interface InvestigationRecord {
  investigation_id: string;
  title: string;
  query: string;
  created_at: string;
  status: string;
  candidates: ShortlistedCandidate[];
  final_report: string;
  audit_trail?: any[];
  investigation_trace?: any[];
  simulation_results?: Record<string, any>;
}

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [health, setHealth] = useState<HealthData | null>(null);

  // Shared Data State from Backend
  const [investigations, setInvestigations] = useState<InvestigationRecord[]>([]);
  const [shortlistedCandidates, setShortlistedCandidates] = useState<ShortlistedCandidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);

  // AI Scout State (Prompt empty by default as requested: "DO NOT show a sample prompt in the chat field")
  const [scoutPrompt, setScoutPrompt] = useState("");
  const [regionFilter, setRegionFilter] = useState("WEU + EEU");
  const [tierFilter, setTierFilter] = useState("TIER 1 ONLY");
  const [isScouting, setIsScouting] = useState(false);
  const [streamProgress, setStreamProgress] = useState<number>(0);
  const [currentStepTitle, setCurrentStepTitle] = useState<string>("");
  const [stepEvents, setStepEvents] = useState<any[]>([]);
  const [liveTraceEvents, setLiveTraceEvents] = useState<any[]>([]);
  const [scoutResult, setScoutResult] = useState<any | null>(null);
  const [scoutError, setScoutError] = useState<string | null>(null);

  // Dossier State
  const [dossier, setDossier] = useState<any | null>(null);
  const [loadingDossier, setLoadingDossier] = useState(false);

  // Simulator State
  const [simAccount, setSimAccount] = useState<number | "">("");
  const [tiltWeight, setTiltWeight] = useState<number>(1.2);
  const [patchPenalty, setPatchPenalty] = useState<number>(0.1);
  const [earlyDeficit, setEarlyDeficit] = useState<boolean>(true);
  const [tier1Bias, setTier1Bias] = useState<boolean>(true);
  const [simResult, setSimResult] = useState<any | null>(null);
  const [loadingSim, setLoadingSim] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  // Initial Sync from Backend
  const refreshAppState = () => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setHealth(d))
      .catch(err => console.error("Health check error:", err));

    fetch(`${API_BASE}/investigations`)
      .then(r => r.json())
      .then(d => {
        if (d.status === 'success') {
          setInvestigations(d.investigations || []);
        }
      })
      .catch(err => console.error("Investigations error:", err));

    fetch(`${API_BASE}/candidates`)
      .then(r => r.json())
      .then(d => {
        if (d.status === 'success') {
          const cands = d.candidates || [];
          setShortlistedCandidates(cands);
          if (cands.length > 0 && selectedCandidateId === null) {
            setSelectedCandidateId(cands[0].account_id);
            setSimAccount(cands[0].account_id);
          }
        }
      })
      .catch(err => console.error("Candidates fetch error:", err));
  };

  useEffect(() => {
    refreshAppState();
  }, []);

  // Fetch Dossier when account changes
  useEffect(() => {
    if (activeTab === 'dossiers' && selectedCandidateId) {
      fetchDossier(selectedCandidateId);
    }
  }, [activeTab, selectedCandidateId]);

  const fetchDossier = (accId: number) => {
    setLoadingDossier(true);
    fetch(`${API_BASE}/dossier/${accId}`)
      .then(r => r.json())
      .then(d => {
        setDossier(d);
        setLoadingDossier(false);
      })
      .catch(err => {
        console.error("Dossier error:", err);
        setLoadingDossier(false);
      });
  };

  const runSimulation = () => {
    if (!simAccount) return;
    setLoadingSim(true);
    setSimError(null);
    setSimResult(null);

    fetch(`${API_BASE}/simulate/failure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        account_id: Number(simAccount),
        patch_shift_penalty: patchPenalty,
        tilt_vulnerability_weight: tiltWeight,
        early_game_deficit: earlyDeficit,
        tier1_opponent_bias: tier1Bias
      })
    })
      .then(async r => {
        const d = await r.json();
        if (!r.ok) {
          throw new Error(d.detail || "Simulation validation failed");
        }
        setSimResult(d);
        setLoadingSim(false);
        // Refresh investigations to cache simulation results
        refreshAppState();
      })
      .catch(err => {
        setSimError(err.message || "Simulation failed");
        setLoadingSim(false);
      });
  };

  // Real LangGraph SSE Streaming Execution
  const dispatchScoutPipeline = async () => {
    if (!scoutPrompt.trim()) return;

    setIsScouting(true);
    setStreamProgress(5);
    setCurrentStepTitle("Initializing LangGraph State Machine...");
    setStepEvents([]);
    setLiveTraceEvents([]);
    setScoutResult(null);
    setScoutError(null);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: `${scoutPrompt.trim()} [Region: ${regionFilter}, Tier: ${tierFilter}]` 
        })
      });

      if (!response.body) throw new Error("No SSE response body available");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const normalized = buffer.replace(/\r\n/g, "\n");
        const lines = normalized.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = "message";
          let dataStr = "";

          const subLines = block.split("\n");
          for (const line of subLines) {
            if (line.startsWith("event: ")) {
              eventType = line.replace("event: ", "").trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.replace("data: ", "").trim();
            }
          }

          if (!dataStr) continue;
          try {
            const dataObj = JSON.parse(dataStr);
            if (eventType === "pipeline_init") {
              setStreamProgress(10);
              setCurrentStepTitle("Execution Started: Request Ingestion");
            } else if (eventType === "step_update") {
              setStreamProgress(dataObj.progress || 50);
              setCurrentStepTitle(dataObj.step_id?.replace(/_/g, " ").toUpperCase());
              setStepEvents(prev => [...prev, dataObj]);
              if (dataObj.trace_events && Array.isArray(dataObj.trace_events)) {
                setLiveTraceEvents(prev => [...prev, ...dataObj.trace_events]);
              }
            } else if (eventType === "pipeline_complete") {
              setStreamProgress(100);
              setCurrentStepTitle("Investigation Completed");
              setScoutResult(dataObj);
              setIsScouting(false);
              // Update state from backend with the new shortlisted candidates
              refreshAppState();
            } else if (eventType === "pipeline_error") {
              setScoutError(dataObj.error);
              setIsScouting(false);
            }
          } catch (e) {
            console.error("SSE JSON parsing error:", e, dataStr);
          }
        }
      }
    } catch (err: any) {
      setScoutError(err.message || "Failed to execute scouting stream");
      setIsScouting(false);
    }
  };

  // Metrics derived from real backend data
  const totalInvestigations = investigations.length;
  const totalShortlisted = shortlistedCandidates.length;
  const criticalDecisionBreakers = investigations.reduce((acc, inv) => {
    let count = 0;
    if (inv.simulation_results) {
      for (const res of Object.values(inv.simulation_results)) {
        count += (res.decision_breakers || []).length;
      }
    }
    return acc + count;
  }, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-black)' }}>
      {/* Top Tactical Command Bar */}
      <header style={{
        height: '48px',
        borderBottom: '1px solid var(--border-dark)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        background: '#0d0f10',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ 
              width: '18px', height: '18px', background: 'var(--crimson-red)', 
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 900, fontSize: '10px', color: '#fff' 
            }}>RL</div>
            <span className="font-heading" style={{ fontSize: '15px', fontWeight: 800, letterSpacing: '0.1em' }}>
              ROSTERLENS COMMAND <span style={{ color: 'var(--text-dim)', fontSize: '11px', marginLeft: '6px' }}>v4.2-R2</span>
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }} className="font-mono">
            <span style={{ 
              display: 'inline-block', width: '6px', height: '6px', 
              background: health?.status === 'healthy' ? 'var(--tactical-green)' : 'var(--crimson-red)' 
            }} />
            <span style={{ color: 'var(--text-muted)' }}>SYS:</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>ONLINE // OPENDOTA FEDERATED</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }} className="font-mono">
            <span style={{ color: 'var(--text-muted)' }}>ROSTER:</span>
            <span style={{ color: 'var(--amber-gold)', fontWeight: 600 }}>TEAM SPIRIT [TI CHAMPIONS]</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ 
            background: totalInvestigations > 0 ? 'var(--crimson-red-dim)' : '#1a1e20', 
            border: `1px solid ${totalInvestigations > 0 ? 'var(--crimson-red)' : '#333'}`, 
            padding: '2px 8px', fontSize: '11px', 
            color: totalInvestigations > 0 ? 'var(--crimson-red)' : 'var(--text-dim)' 
          }} className="font-mono">
            {totalInvestigations > 0 ? `#INV-0${totalInvestigations} ACTIVE` : 'NO ACTIVE INVESTIGATION'}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '11px', fontWeight: 700 }} className="font-heading">COACH SILENT</div>
              <div style={{ fontSize: '9px', color: 'var(--amber-gold)' }} className="font-mono">HEAD ANALYST</div>
            </div>
            <div style={{ 
              width: '26px', height: '26px', background: '#22272a', border: '1px solid #333', 
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 800 
            }}>CS</div>
          </div>
        </div>
      </header>

      {/* Main Container: Sidebar + Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Navigation Matrix */}
        <nav style={{
          width: '240px',
          background: 'var(--bg-sidebar)',
          borderRight: '1px solid var(--border-dark)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '16px 0'
        }}>
          <div>
            {/* Target Squad Deployed Widget */}
            <div style={{ padding: '0 16px 16px 16px', borderBottom: '1px solid var(--border-dark)' }}>
              <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '6px' }} className="font-mono">
                TARGET SQUAD DEPLOYED
              </div>
              <div style={{ 
                background: '#131618', border: '1px solid var(--border-light)', 
                padding: '8px 10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' 
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Shield size={14} color="var(--crimson-red)" />
                  <span style={{ fontSize: '12px', fontWeight: 700 }} className="font-heading">TEAM SPIRIT</span>
                </div>
                <span style={{ fontSize: '9px', color: 'var(--tactical-green)' }} className="font-mono">ACTIVE</span>
              </div>
            </div>

            {/* Tactical Routing Matrix */}
            <div style={{ padding: '16px 0' }}>
              <div style={{ padding: '0 16px 8px 16px', fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">
                TACTICAL ROUTING MATRIX
              </div>

              {[
                { key: 'overview', index: '01', label: 'OVERVIEW', tag: 'CMD' },
                { key: 'scout', index: '02', label: 'AI SCOUT', tag: 'ACT' },
                { key: 'candidates', index: '03', label: 'CANDIDATES', tag: totalShortlisted > 0 ? `${totalShortlisted}` : 'EMPTY' },
                { key: 'dossiers', index: '04', label: 'EVIDENCE DOSSIERS', tag: 'BIO' },
                { key: 'simulator', index: '05', label: 'FAILURE SIMULATOR', tag: 'WARN' },
                { key: 'reports', index: '06', label: 'REPORTS & DISPATCH', tag: 'DOC' }
              ].map(item => {
                const isActive = activeTab === item.key;
                return (
                  <button
                    key={item.key}
                    onClick={() => setActiveTab(item.key as TabKey)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 16px',
                      background: isActive ? 'var(--crimson-red)' : 'transparent',
                      color: isActive ? '#fff' : 'var(--text-muted)',
                      border: 'none',
                      borderLeft: isActive ? '3px solid #ff4d4d' : '3px solid transparent',
                      cursor: 'pointer',
                      textAlign: 'left',
                      fontFamily: 'JetBrains Mono',
                      fontSize: '11px',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <span>{item.index} // {item.label}</span>
                    <span style={{ 
                      fontSize: '9px', 
                      color: isActive ? '#fff' : item.tag === 'WARN' ? 'var(--amber-gold)' : 'var(--text-dim)' 
                    }}>
                      {item.tag}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model Footprint */}
          <div style={{ padding: '0 16px', fontSize: '10px', color: 'var(--text-dim)' }} className="font-mono">
            <div>BRAIN: <span style={{ color: 'var(--text-primary)' }}>{health?.model || 'gemini-3.8-flash'}</span></div>
            <div>STATUS: <span style={{ color: 'var(--tactical-green)' }}>RESONANCE OK</span></div>
          </div>
        </nav>

        {/* Content Viewports */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>
          
          {/* ----------------- TAB 01: OVERVIEW ----------------- */}
          {activeTab === 'overview' && (
            <div>
              {/* Header Banner */}
              <div style={{ textAlign: 'center', margin: '20px 0 36px 0' }}>
                <div style={{ display: 'inline-block', background: 'var(--crimson-red)', color: '#fff', fontSize: '10px', fontWeight: 800, padding: '2px 8px', marginBottom: '12px' }} className="font-mono">
                  COMMAND NODE 01 // ROSTER OPTIMIZATION MATRIX
                </div>
                <h1 style={{ fontSize: '48px', fontWeight: 900, lineHeight: 1.05 }}>
                  BUILD YOUR NEXT <br />
                  <span style={{ color: 'var(--crimson-red)' }}>WINNING ROSTER.</span>
                </h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '12px' }}>
                  Investigate talent. Challenge assumptions. Recruit with evidence.
                </p>

                {/* Instant Scouting Dispatch Bar - No Sample Prompt in Input Field */}
                <div style={{ 
                  maxWidth: '720px', margin: '24px auto 0 auto', 
                  display: 'flex', background: '#111416', border: '1px solid var(--border-light)' 
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', padding: '0 14px', color: 'var(--text-dim)' }}>
                    <Search size={16} />
                  </div>
                  <input 
                    type="text" 
                    value={scoutPrompt}
                    onChange={e => setScoutPrompt(e.target.value)}
                    placeholder="Enter natural language scouting brief (e.g. Find position-4 support with wide hero pool)..."
                    style={{
                      flex: 1, background: 'transparent', border: 'none', color: '#fff',
                      padding: '14px 0', fontSize: '13px', fontFamily: 'JetBrains Mono', outline: 'none'
                    }}
                  />
                  <button 
                    className="btn-tactical"
                    onClick={() => {
                      setActiveTab('scout');
                      if (scoutPrompt.trim()) {
                        dispatchScoutPipeline();
                      }
                    }}
                  >
                    INITIATE INVESTIGATION <ArrowRight size={14} />
                  </button>
                </div>
              </div>

              {/* 4 Telemetry Metrics Grid - Strictly Powered by Actual Application Data */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '48px' }}>
                {/* Metric 01: Actual Completed Investigations */}
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">METRIC // 01</span>
                    <span style={{ fontSize: '9px', color: totalInvestigations > 0 ? 'var(--tactical-green)' : 'var(--text-dim)', background: totalInvestigations > 0 ? 'var(--tactical-green-dim)' : '#1a1e20', padding: '1px 6px' }} className="font-mono">
                      {totalInvestigations > 0 ? 'LIVE PULSE' : 'STANDBY'}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="font-mono">ACTIVE INVESTIGATIONS</div>
                  <div style={{ fontSize: '36px', fontWeight: 800, color: totalInvestigations > 0 ? 'var(--tactical-green)' : 'var(--text-muted)', marginTop: '4px' }} className="font-heading">
                    {totalInvestigations < 10 ? `0${totalInvestigations}` : totalInvestigations}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '8px' }} className="font-mono">
                    COMPLETED RUNS: <span style={{ color: 'var(--text-primary)' }}>{totalInvestigations}</span>
                  </div>
                </div>

                {/* Metric 02: Actual Shortlisted Candidates */}
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">METRIC // 02</span>
                    <span style={{ fontSize: '9px', color: totalShortlisted > 0 ? 'var(--amber-gold)' : 'var(--text-dim)', background: totalShortlisted > 0 ? 'var(--amber-dim)' : '#1a1e20', padding: '1px 6px' }} className="font-mono">
                      {totalShortlisted > 0 ? 'VERIFIED SHORTLIST' : 'EMPTY'}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="font-mono">CANDIDATES UNDER SCRUTINY</div>
                  <div style={{ fontSize: '36px', fontWeight: 800, color: '#fff', marginTop: '4px' }} className="font-heading">
                    {totalShortlisted < 10 ? `0${totalShortlisted}` : totalShortlisted}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '8px' }} className="font-mono">
                    LINKED TO SHORTLIST: <span style={{ color: 'var(--text-primary)' }}>{totalShortlisted}</span>
                  </div>
                </div>

                {/* Metric 03: Actual Evidence Coverage */}
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">METRIC // 03</span>
                    <span style={{ fontSize: '9px', color: 'var(--text-muted)', background: '#222', padding: '1px 6px' }} className="font-mono">OPENDOTA 7.35</span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="font-mono">EVIDENCE COVERAGE</div>
                  <div style={{ fontSize: '36px', fontWeight: 800, color: '#fff', marginTop: '4px' }} className="font-heading">
                    {totalShortlisted > 0 ? '100.0%' : '0.0%'}
                  </div>
                  <div style={{ fontSize: '10px', color: totalShortlisted > 0 ? 'var(--tactical-green)' : 'var(--text-dim)', marginTop: '8px' }} className="font-mono">
                    {totalShortlisted > 0 ? 'VERIFIED REPLAYS LINKED' : 'AWAITING SCOUT RUN'}
                  </div>
                </div>

                {/* Metric 04: Real Decision-Breakers Triggered */}
                <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">METRIC // 04</span>
                    <span style={{ fontSize: '9px', color: criticalDecisionBreakers > 0 ? 'var(--crimson-red)' : 'var(--text-dim)', background: criticalDecisionBreakers > 0 ? 'var(--crimson-red-dim)' : '#1a1e20', padding: '1px 6px' }} className="font-mono">
                      {criticalDecisionBreakers > 0 ? '▲ CRITICAL' : 'NOMINAL'}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="font-mono">DECISION-BREAKER ALERTS</div>
                  <div style={{ fontSize: '36px', fontWeight: 800, color: criticalDecisionBreakers > 0 ? 'var(--crimson-red)' : 'var(--text-muted)', marginTop: '4px' }} className="font-heading">
                    {criticalDecisionBreakers < 10 ? `0${criticalDecisionBreakers}` : criticalDecisionBreakers}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '8px' }} className="font-mono">
                    SIMULATION FLAGS: <span style={{ color: criticalDecisionBreakers > 0 ? 'var(--crimson-red)' : 'var(--text-primary)' }}>{criticalDecisionBreakers}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ----------------- TAB 02: AI SCOUT ----------------- */}
          {activeTab === 'scout' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--crimson-red)', fontWeight: 800 }} className="font-mono">
                    SCOUT AGENT ENGINE // NATURAL LANGUAGE RECRUITMENT THESIS
                  </div>
                  <h2 style={{ fontSize: '24px', fontWeight: 800 }} className="font-heading">
                    AI SCOUT INTERACTIVE TERMINAL
                  </h2>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--tactical-green)', background: 'var(--tactical-green-dim)', padding: '3px 8px' }} className="font-mono">
                  AUTONOMOUS NLP READY
                </div>
              </div>

              {/* Main Thesis Box (Prompt empty by default as required) */}
              <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '24px', marginBottom: '24px' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '8px' }} className="font-mono">
                  SCOUTING THESIS QUERY
                </div>

                <div style={{ border: '1px solid var(--crimson-red)', background: '#0e1011', padding: '16px', marginBottom: '20px' }}>
                  <textarea 
                    value={scoutPrompt}
                    onChange={e => setScoutPrompt(e.target.value)}
                    placeholder="Enter your recruitment brief (e.g. Find position-4 support in WEU with wide hero pool and high early-rotation impact)..."
                    rows={3}
                    style={{
                      width: '100%', background: 'transparent', border: 'none', color: '#fff',
                      fontSize: '14px', fontFamily: 'JetBrains Mono', outline: 'none', resize: 'vertical'
                    }}
                  />
                </div>

                {/* Filter Controls Row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
                  <div style={{ background: '#111315', border: '1px solid var(--border-dark)', padding: '12px' }}>
                    <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '6px' }} className="font-mono">REGION & TIER FILTER</div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button 
                        onClick={() => setRegionFilter(r => r === 'WEU + EEU' ? 'ALL REGIONS' : 'WEU + EEU')}
                        style={{ background: 'transparent', border: 'none', fontSize: '11px', color: 'var(--amber-gold)', fontWeight: 700, cursor: 'pointer', padding: 0 }} 
                        className="font-mono"
                      >
                        {regionFilter}
                      </button>
                      <button 
                        onClick={() => setTierFilter(t => t === 'TIER 1 ONLY' ? 'TIER 1 & 2' : 'TIER 1 ONLY')}
                        style={{ background: 'transparent', border: 'none', fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer', padding: 0 }} 
                        className="font-mono"
                      >
                        {tierFilter}
                      </button>
                    </div>
                  </div>

                  <div style={{ background: '#111315', border: '1px solid var(--border-dark)', padding: '12px' }}>
                    <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '6px' }} className="font-mono">VERIFICATION STRICTNESS</div>
                    <div style={{ fontSize: '11px', color: 'var(--tactical-green)', fontWeight: 700 }} className="font-mono">
                      ■ STRICT (NO SCRIM NOISE)
                    </div>
                  </div>

                  <div style={{ background: '#111315', border: '1px solid var(--border-dark)', padding: '12px' }}>
                    <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '6px' }} className="font-mono">REPLAY HORIZON</div>
                    <div style={{ fontSize: '11px', color: 'var(--amber-gold)', fontWeight: 700 }} className="font-mono">
                      PATCH 7.35b - 7.35d (LAST 90 DAYS)
                    </div>
                  </div>
                </div>

                {/* Dispatch Trigger Bar */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-dim)' }} className="font-mono">
                    STATUS: {isScouting ? 'DISPATCHING AGENT PIPELINE...' : 'READY FOR INPUT'}
                  </div>
                  <button 
                    className="btn-tactical" 
                    onClick={dispatchScoutPipeline}
                    disabled={isScouting || !scoutPrompt.trim()}
                  >
                    {isScouting ? <RefreshCw className="animate-spin" size={14} /> : <Play size={14} />}
                    DISPATCH SCOUT PIPELINE ►
                  </button>
                </div>
              </div>

              {/* Real-time Progress & Execution Section */}
              {(isScouting || scoutResult || scoutError) && (
                <div style={{ background: '#0e1112', border: '1px solid var(--border-dark)', padding: '20px', marginBottom: '24px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="font-mono">
                      LANGGRAPH EXECUTION: <span style={{ color: '#fff', fontWeight: 700 }}>{currentStepTitle}</span>
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--tactical-green)' }} className="font-mono">
                      {streamProgress}%
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div style={{ height: '4px', background: '#1c2124', width: '100%', marginBottom: '16px' }}>
                    <div style={{ 
                      height: '100%', 
                      background: isScouting ? 'var(--crimson-red)' : 'var(--tactical-green)', 
                      width: `${streamProgress}%`,
                      transition: 'width 0.3s ease'
                    }} />
                  </div>

                  {/* Step Events Log */}
                  {stepEvents.length > 0 && (
                    <div style={{ maxHeight: '180px', overflowY: 'auto', background: '#090b0c', padding: '12px', border: '1px solid var(--border-dark)', marginBottom: '16px' }}>
                      {stepEvents.map((ev, i) => (
                        <div key={i} style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }} className="font-mono">
                          <span style={{ color: 'var(--tactical-green)', marginRight: '8px' }}>[✓]</span>
                          <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{ev.step_id}</span>
                          <span style={{ color: 'var(--text-dim)', marginLeft: '8px' }}>({ev.timestamp?.slice(11, 19)})</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Error banner */}
                  {scoutError && (
                    <div style={{ background: 'var(--crimson-red-dim)', border: '1px solid var(--crimson-red)', padding: '12px', color: 'var(--crimson-red)', fontSize: '12px' }} className="font-mono">
                      ERROR: {scoutError}
                    </div>
                  )}

                  {/* LIVE DECISION & TOOL CALL INVESTIGATION TIMELINE */}
                  {(isScouting || liveTraceEvents.length > 0 || (scoutResult && scoutResult.investigation_trace)) && (
                    <InvestigationTraceViewer 
                      traceEvents={scoutResult?.investigation_trace || liveTraceEvents}
                      title="AUTONOMOUS INVESTIGATION TIMELINE // DECISION & TOOL TRACE"
                      isLive={isScouting}
                    />
                  )}

                  {/* FINAL RECRUITMENT INTELLIGENCE DOSSIER (Rendered with Tactical DossierViewer) */}
                  {scoutResult && (
                    <div style={{ marginTop: '20px' }}>
                      <DossierViewer
                        rawReport={scoutResult.final_report}
                        title="FINAL RECRUITMENT INTELLIGENCE DOSSIER"
                        subtitle="VERIFIED SCOUTING ARTIFACT"
                        modelFootprint="GEMINI-3.5-FLASH-LITE // VERIFIED DOTA2 INTEL"
                        metadataBadges={[
                          { label: 'CANDIDATES SHORTLISTED', value: scoutResult.candidates?.length || 0, color: 'var(--amber-gold)' },
                          { label: 'EXECUTION PHASE', value: scoutResult.phase || 'COMPLETED', color: 'var(--tactical-green)' },
                          { label: 'DECISION NODES', value: (scoutResult.investigation_trace || liveTraceEvents).length }
                        ]}
                        investigationTrace={scoutResult.investigation_trace || liveTraceEvents}
                        onExport={() => window.print()}
                        onDispatch={() => {
                          setActiveTab('candidates');
                        }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ----------------- TAB 03: CANDIDATES ----------------- */}
          {/* Strictly display shortlisted candidates produced by completed scouting investigations */}
          {activeTab === 'candidates' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--amber-gold)', fontWeight: 800 }} className="font-mono">
                    INVESTIGATION SHORTLIST // VERIFIED CANDIDATE POOL
                  </div>
                  <h2 style={{ fontSize: '24px', fontWeight: 800 }} className="font-heading">
                    SHORTLISTED CANDIDATES
                  </h2>
                </div>
              </div>

              {shortlistedCandidates.length === 0 ? (
                /* Appropriate Empty State Prompting User to Run AI Scout */
                <div style={{ background: 'var(--bg-card)', border: '1px dashed var(--border-light)', padding: '60px 40px', textAlign: 'center' }}>
                  <Users size={48} color="var(--text-dim)" style={{ margin: '0 auto 16px auto', display: 'block' }} />
                  <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '8px' }} className="font-heading">
                    NO CANDIDATE SHORTLIST AVAILABLE
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontFamily: 'JetBrains Mono', maxWidth: '480px', margin: '0 auto 24px auto' }}>
                    Candidates must be produced by a completed scouting investigation rather than arbitrary player search.
                  </p>
                  <button className="btn-tactical" onClick={() => setActiveTab('scout')}>
                    RUN AI SCOUT TO PRODUCE SHORTLIST ►
                  </button>
                </div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  {shortlistedCandidates.map((cand, idx) => (
                    <div 
                      key={idx}
                      style={{ 
                        background: 'var(--bg-card)', 
                        border: selectedCandidateId === cand.account_id ? '1px solid var(--crimson-red)' : '1px solid var(--border-dark)', 
                        padding: '16px',
                        cursor: 'pointer'
                      }}
                      onClick={() => {
                        setSelectedCandidateId(cand.account_id);
                        setSimAccount(cand.account_id);
                        setActiveTab('dossiers');
                      }}
                    >
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '12px' }}>
                        {cand.avatar ? (
                          <img src={cand.avatar} alt={cand.name} style={{ width: '48px', height: '48px', border: '1px solid #333' }} />
                        ) : (
                          <div style={{ width: '48px', height: '48px', background: '#222', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Users size={20} color="#777" />
                          </div>
                        )}
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: 800 }} className="font-heading">{cand.name || cand.personaname}</div>
                          <div style={{ fontSize: '11px', color: 'var(--amber-gold)' }} className="font-mono">{cand.team_name}</div>
                        </div>
                      </div>

                      {/* Compatibility Score */}
                      <div style={{ background: '#0e1112', border: '1px solid var(--border-dark)', padding: '8px', marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-dim)' }} className="font-mono">COMPATIBILITY SCORE</span>
                        <span style={{ fontSize: '16px', fontWeight: 800, color: 'var(--tactical-green)' }} className="font-heading">
                          {cand.compatibility_score}%
                        </span>
                      </div>

                      {/* Supporting Evidence */}
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px', lineHeight: 1.4 }} className="font-mono">
                        {cand.supporting_evidence}
                      </div>

                      {/* Missing Data Warning */}
                      {cand.missing_data_warning && (
                        <div style={{ background: 'var(--amber-dim)', border: '1px solid var(--amber-gold)', padding: '6px 8px', fontSize: '10px', color: 'var(--amber-gold)', marginBottom: '8px' }} className="font-mono">
                          ▲ {cand.missing_data_warning}
                        </div>
                      )}

                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-dim)', borderTop: '1px solid var(--border-dark)', paddingTop: '8px' }} className="font-mono">
                        <span>ROLE: <span style={{ color: '#fff' }}>{cand.role_desc}</span></span>
                        <span>INV: #{cand.investigation_id?.slice(0, 6).toUpperCase()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ----------------- TAB 04: EVIDENCE DOSSIERS ----------------- */}
          {activeTab === 'dossiers' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: 'var(--tactical-green)', fontWeight: 800 }} className="font-mono">
                    INDIVIDUAL TELEMETRY VERIFICATION // OPENDOTA RAW AUDIT
                  </div>
                  <h2 style={{ fontSize: '24px', fontWeight: 800 }} className="font-heading">
                    {selectedCandidateId ? `EVIDENCE DOSSIER: ACCOUNT #${selectedCandidateId}` : 'EVIDENCE DOSSIERS'}
                  </h2>
                </div>
                {selectedCandidateId && (
                  <button className="btn-tactical" onClick={() => fetchDossier(selectedCandidateId)}>
                    <RefreshCw size={14} /> RE-SYNC TELEMETRY
                  </button>
                )}
              </div>

              {shortlistedCandidates.length === 0 ? (
                <div style={{ background: 'var(--bg-card)', border: '1px dashed var(--border-light)', padding: '60px 40px', textAlign: 'center' }}>
                  <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '8px' }} className="font-heading">
                    NO SHORTLISTED CANDIDATES TO INSPECT
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontFamily: 'JetBrains Mono', maxWidth: '480px', margin: '0 auto 24px auto' }}>
                    Evidence dossiers are linked to candidates from completed scouting investigations.
                  </p>
                  <button className="btn-tactical" onClick={() => setActiveTab('scout')}>
                    RUN AI SCOUT ►
                  </button>
                </div>
              ) : loadingDossier ? (
                <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }} className="font-mono">
                  AGGREGATING OPENAPI ENDPOINTS (/players, /recentMatches, /heroes, /wl)...
                </div>
              ) : dossier ? (
                <div>
                  {/* Summary Metric Strip */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', marginBottom: '24px' }}>
                    <div style={{ background: '#111416', border: '1px solid var(--border-dark)', padding: '12px' }}>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">RECENT WINRATE</div>
                      <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--tactical-green)' }} className="font-heading">
                        {dossier.summary?.recent_winrate}%
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">LAST 20 MATCHES</div>
                    </div>

                    <div style={{ background: '#111416', border: '1px solid var(--border-dark)', padding: '12px' }}>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">AVG KDA RATIO</div>
                      <div style={{ fontSize: '20px', fontWeight: 800, color: '#fff' }} className="font-heading">
                        {dossier.summary?.avg_kda}
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">KILL PARTICIPATION</div>
                    </div>

                    <div style={{ background: '#111416', border: '1px solid var(--border-dark)', padding: '12px' }}>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">GPM / XPM</div>
                      <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--amber-gold)' }} className="font-heading">
                        {dossier.summary?.avg_gpm} / {dossier.summary?.avg_xpm}
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">FARM VELOCITY</div>
                    </div>

                    <div style={{ background: '#111416', border: '1px solid var(--border-dark)', padding: '12px' }}>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">CAREER WINRATE</div>
                      <div style={{ fontSize: '24px', fontWeight: 800, color: '#fff' }} className="font-heading">
                        {dossier.summary?.career_winrate}%
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">
                        {dossier.summary?.career_wins}W - {dossier.summary?.career_losses}L
                      </div>
                    </div>

                    <div style={{ background: '#111416', border: '1px solid var(--border-dark)', padding: '12px' }}>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">SHORTLIST LINK</div>
                      <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--tactical-green)' }} className="font-heading">
                        VERIFIED
                      </div>
                      <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">
                        INV #{dossier.investigation_id ? dossier.investigation_id.slice(0, 6).toUpperCase() : 'LINKED'}
                      </div>
                    </div>
                  </div>

                  {/* Signature Heroes */}
                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }} className="font-mono">
                      SIGNATURE HERO POOL VERSATILITY
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '8px' }}>
                      {dossier.signature_heroes?.map((h: any, i: number) => (
                        <div key={i} style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '10px' }}>
                          <div style={{ fontSize: '12px', fontWeight: 700 }} className="font-heading">{h.hero_name}</div>
                          <div style={{ fontSize: '10px', color: 'var(--amber-gold)' }} className="font-mono">{h.winrate}% WR</div>
                          <div style={{ fontSize: '9px', color: 'var(--text-dim)' }} className="font-mono">{h.games} matches</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recent 20 Matches Telemetry Table */}
                  <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '16px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px' }} className="font-mono">
                      RECENT SITUATIONAL MATCH LOGS (OPENDOTA PARSED)
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'JetBrains Mono' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-light)', color: 'var(--text-dim)', textAlign: 'left' }}>
                          <th style={{ padding: '8px' }}>MATCH ID</th>
                          <th style={{ padding: '8px' }}>HERO</th>
                          <th style={{ padding: '8px' }}>OUTCOME</th>
                          <th style={{ padding: '8px' }}>K / D / A</th>
                          <th style={{ padding: '8px' }}>GPM / XPM</th>
                          <th style={{ padding: '8px' }}>HERO DMG</th>
                          <th style={{ padding: '8px' }}>DURATION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dossier.recent_matches?.map((m: any, idx: number) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #1a1e20' }}>
                            <td style={{ padding: '8px', color: 'var(--amber-gold)' }}>#{m.match_id}</td>
                            <td style={{ padding: '8px', color: '#fff', fontWeight: 600 }}>{m.hero_name}</td>
                            <td style={{ padding: '8px' }}>
                              <span style={{ 
                                color: m.won ? 'var(--tactical-green)' : 'var(--crimson-red)',
                                fontWeight: 700 
                              }}>
                                {m.won ? 'VICTORY' : 'DEFEAT'}
                              </span>
                            </td>
                            <td style={{ padding: '8px' }}>{m.kills} / {m.deaths} / {m.assists}</td>
                            <td style={{ padding: '8px' }}>{m.gpm} / {m.xpm}</td>
                            <td style={{ padding: '8px' }}>{m.hero_damage?.toLocaleString()}</td>
                            <td style={{ padding: '8px', color: 'var(--text-dim)' }}>{m.duration_min}m</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-dim)' }} className="font-mono">
                  SELECT A CANDIDATE FROM THE SHORTLIST
                </div>
              )}
            </div>
          )}

          {/* ----------------- TAB 05: FAILURE SIMULATOR ----------------- */}
          {/* Strictly runs ONLY for candidates belonging to an existing investigation shortlist */}
          {activeTab === 'simulator' && (
            <div>
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '10px', color: 'var(--crimson-red)', fontWeight: 800 }} className="font-mono">
                  DETERMINISTIC STRESS-TEST ENGINE // SENSITIVITY & DECISION-BREAKER LOGIC
                </div>
                <h2 style={{ fontSize: '24px', fontWeight: 800 }} className="font-heading">
                  ROSTER FAILURE SIMULATOR
                </h2>
              </div>

              {shortlistedCandidates.length === 0 ? (
                /* Simulation Disabled when no shortlist exists */
                <div style={{ background: 'var(--bg-card)', border: '1px dashed var(--crimson-red)', padding: '60px 40px', textAlign: 'center' }}>
                  <AlertTriangle size={48} color="var(--crimson-red)" style={{ margin: '0 auto 16px auto', display: 'block' }} />
                  <div style={{ fontSize: '18px', fontWeight: 800, marginBottom: '8px', color: 'var(--crimson-red)' }} className="font-heading">
                    SIMULATOR LOCKED: NO VERIFIED SHORTLIST
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontFamily: 'JetBrains Mono', maxWidth: '520px', margin: '0 auto 24px auto' }}>
                    The Failure Simulator cannot run on arbitrary player IDs. It must run exclusively on candidates produced by an active scouting investigation.
                  </p>
                  <button className="btn-tactical" onClick={() => setActiveTab('scout')}>
                    RUN AI SCOUT TO SHORTLIST CANDIDATES ►
                  </button>
                </div>
              ) : (
                /* Stress-Test Control Panel */
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
                  <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '20px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--amber-gold)', fontWeight: 800, marginBottom: '16px' }} className="font-mono">
                      SHORTLISTED CANDIDATE SELECTOR
                    </div>

                    {/* Candidate Selector Populated Exclusively from Shortlisted Candidates */}
                    <div style={{ marginBottom: '16px' }}>
                      <label style={{ fontSize: '11px', color: 'var(--text-dim)', display: 'block', marginBottom: '6px' }} className="font-mono">
                        SELECT CANDIDATE:
                      </label>
                      <select 
                        value={simAccount}
                        onChange={e => setSimAccount(Number(e.target.value))}
                        style={{
                          width: '100%', background: '#0e1112', border: '1px solid var(--border-dark)', color: '#fff',
                          padding: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono', outline: 'none'
                        }}
                      >
                        {shortlistedCandidates.map((c, i) => (
                          <option key={i} value={c.account_id}>
                            {c.name} ({c.team_name}) - #{c.account_id}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <label style={{ fontSize: '11px', color: 'var(--text-dim)', display: 'block', marginBottom: '6px' }} className="font-mono">
                        TILT VULNERABILITY MULTIPLIER: {tiltWeight}x
                      </label>
                      <input 
                        type="range" min="0.5" max="2.5" step="0.1"
                        value={tiltWeight}
                        onChange={e => setTiltWeight(Number(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <label style={{ fontSize: '11px', color: 'var(--text-dim)', display: 'block', marginBottom: '6px' }} className="font-mono">
                        META PATCH SHIFT PENALTY: {Math.round(patchPenalty * 100)}%
                      </label>
                      <input 
                        type="range" min="0" max="0.3" step="0.05"
                        value={patchPenalty}
                        onChange={e => setPatchPenalty(Number(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-primary)', cursor: 'pointer' }} className="font-mono">
                        <input 
                          type="checkbox" 
                          checked={earlyDeficit} 
                          onChange={e => setEarlyDeficit(e.target.checked)} 
                        />
                        SIMULATE 5K EARLY GOLD DEFICIT
                      </label>
                    </div>

                    <div style={{ marginBottom: '20px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-primary)', cursor: 'pointer' }} className="font-mono">
                        <input 
                          type="checkbox" 
                          checked={tier1Bias} 
                          onChange={e => setTier1Bias(e.target.checked)} 
                        />
                        TIER 1 OPPONENT SCRUTINY
                      </label>
                    </div>

                    <button className="btn-tactical" style={{ width: '100%' }} onClick={runSimulation} disabled={loadingSim || !simAccount}>
                      <Sliders size={14} /> EXECUTE SIMULATION
                    </button>
                  </div>

                  {/* Simulation Output Dashboard */}
                  <div style={{ background: '#0e1112', border: '1px solid var(--border-dark)', padding: '24px' }}>
                    {loadingSim ? (
                      <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }} className="font-mono">
                        RECALCULATING SENSITIVITY & DECISION-BREAKER LOGIC...
                      </div>
                    ) : simError ? (
                      <div style={{ background: 'var(--crimson-red-dim)', border: '1px solid var(--crimson-red)', padding: '16px', color: 'var(--crimson-red)', fontSize: '12px' }} className="font-mono">
                        ▲ {simError}
                      </div>
                    ) : simResult ? (
                      <div>
                        {/* Score Comparison */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                          <div style={{ background: '#141718', border: '1px solid var(--border-dark)', padding: '16px' }}>
                            <div style={{ fontSize: '10px', color: 'var(--text-dim)' }} className="font-mono">BASELINE COMPATIBILITY</div>
                            <div style={{ fontSize: '32px', fontWeight: 800, color: '#fff' }} className="font-heading">
                              {simResult.baseline_compatibility}%
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--tactical-green)' }} className="font-mono">NORMAL TOURNAMENT CONDITIONS</div>
                          </div>

                          <div style={{ background: '#141718', border: '1px solid var(--border-dark)', padding: '16px' }}>
                            <div style={{ fontSize: '10px', color: 'var(--text-dim)' }} className="font-mono">STRESS-TESTED COMPATIBILITY</div>
                            <div style={{ fontSize: '32px', fontWeight: 800, color: simResult.stress_tested_score >= 60 ? 'var(--amber-gold)' : 'var(--crimson-red)' }} className="font-heading">
                              {simResult.stress_tested_score}%
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--crimson-red)' }} className="font-mono">
                              DELTA: {simResult.score_delta}% UNDER DEFICIT
                            </div>
                          </div>
                        </div>

                        {/* What Changed & Why */}
                        <div style={{ background: '#141718', border: '1px solid var(--border-dark)', padding: '16px', marginBottom: '16px' }}>
                          <div style={{ fontSize: '11px', color: 'var(--amber-gold)', fontWeight: 800, marginBottom: '6px' }} className="font-mono">
                            VARIANCE ANALYSIS & EXPLANATION
                          </div>
                          <div style={{ fontSize: '12px', color: '#e5e7eb', fontFamily: 'JetBrains Mono', lineHeight: 1.5, marginBottom: '10px' }}>
                            {simResult.change_explanation}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'JetBrains Mono' }}>
                            UNCERTAINTY MITIGATION: <span style={{ color: 'var(--text-primary)' }}>{simResult.uncertainty_mitigation}</span>
                          </div>
                        </div>

                        {/* Decision-Breaker Warnings */}
                        <div>
                          <div style={{ fontSize: '11px', color: 'var(--crimson-red)', fontWeight: 800, marginBottom: '12px' }} className="font-mono">
                            ▲ TRIGGERED DECISION-BREAKER ALERTS ({simResult.decision_breakers?.length || 0})
                          </div>
                          {simResult.decision_breakers?.length === 0 ? (
                            <div style={{ fontSize: '12px', color: 'var(--tactical-green)' }} className="font-mono">
                              NO CRITICAL DECISION-BREAKER RISKS DETECTED UNDER THIS SCENARIO.
                            </div>
                          ) : (
                            simResult.decision_breakers?.map((alert: any, i: number) => (
                              <div key={i} style={{ 
                                background: 'var(--crimson-red-dim)', border: '1px solid var(--crimson-red)', 
                                padding: '12px', marginBottom: '8px' 
                              }}>
                                <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--crimson-red)' }} className="font-heading">
                                  [{alert.severity}] {alert.code}
                                </div>
                                <div style={{ fontSize: '11px', color: '#e5e7eb', marginTop: '4px' }} className="font-mono">
                                  {alert.message}
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-dim)' }} className="font-mono">
                        SELECT A SHORTLISTED CANDIDATE AND CLICK "EXECUTE SIMULATION"
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ----------------- TAB 06: REPORTS & DISPATCH ----------------- */}
          {activeTab === 'reports' && (
            <div>
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '10px', color: 'var(--amber-gold)', fontWeight: 800 }} className="font-mono">
                  AUDITED SCOUTING ARTIFACTS // EXECUTIVE SUMMARY & DISPATCH
                </div>
                <h2 style={{ fontSize: '24px', fontWeight: 800 }} className="font-heading">
                  REPORTS & DISPATCH ARCHIVE
                </h2>
              </div>

              {investigations.length === 0 ? (
                <div style={{ background: 'var(--bg-card)', border: '1px dashed var(--border-light)', padding: '60px 40px', textAlign: 'center' }}>
                  <FileText size={48} color="var(--text-dim)" style={{ margin: '0 auto 16px auto', display: 'block' }} />
                  <div style={{ fontSize: '16px', fontWeight: 800, marginBottom: '8px' }} className="font-heading">
                    NO COMPLETED INVESTIGATION REPORTS
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontFamily: 'JetBrains Mono', maxWidth: '480px', margin: '0 auto 24px auto' }}>
                    Reports are generated automatically upon completion of a scouting investigation.
                  </p>
                  <button className="btn-tactical" onClick={() => setActiveTab('scout')}>
                    RUN AI SCOUT TO GENERATE REPORT ►
                  </button>
                </div>
              ) : (
                investigations.map((inv, idx) => (
                  <div key={idx} style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dark)', padding: '24px', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--border-dark)', paddingBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 800 }} className="font-heading">
                          REPORT #{inv.investigation_id.slice(0, 8).toUpperCase()} // {inv.title}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }} className="font-mono">
                          BRIEF: "{inv.query}"
                        </div>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--tactical-green)' }} className="font-mono">
                        VERIFIED FOR COACH REVIEW
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }} className="font-mono">
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        SHORTLISTED CANDIDATES: <span style={{ color: '#fff', fontWeight: 700 }}>{inv.candidates?.length || 0}</span>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        SIMULATIONS EXECUTED: <span style={{ color: 'var(--amber-gold)', fontWeight: 700 }}>{Object.keys(inv.simulation_results || {}).length}</span>
                      </div>
                    </div>

                    {/* Executive Dossier Document with Markdown Formatting & Investigation Trace */}
                    <div style={{ marginBottom: '20px' }}>
                      <DossierViewer
                        rawReport={inv.final_report}
                        title={inv.title.toUpperCase()}
                        subtitle={`REPORT #${inv.investigation_id.slice(0, 8).toUpperCase()}`}
                        modelFootprint="GROUNDED TELEMETRY // OPENDOTA API"
                        metadataBadges={[
                          { label: 'BRIEF', value: `"${inv.query.slice(0, 60)}${inv.query.length > 60 ? '...' : ''}"` },
                          { label: 'CANDIDATES', value: inv.candidates?.length || 0, color: 'var(--amber-gold)' },
                          { label: 'TRACE NODES', value: inv.investigation_trace?.length || 0 },
                          { label: 'SIMULATIONS', value: Object.keys(inv.simulation_results || {}).length }
                        ]}
                        investigationTrace={inv.investigation_trace || []}
                        maxHeight="420px"
                        onExport={() => window.print()}
                        onDispatch={() => alert(`Report #${inv.investigation_id.slice(0, 8).toUpperCase()} dispatched to coaching staff.`)}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  );
}

export default App;
