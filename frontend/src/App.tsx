import React, { useState, useEffect, useRef } from "react";

interface PredictionResult {
  prediction: string;
  confidence: number;
  score: number;
  timestamp: string;
  model_name?: string;
  feature_importance?: number[];
  explanation?: string;
  filename: string;
  latency_ms?: number;
}

interface ModelInfo {
  name: string;
  type: string;
  status: string;
}

interface HealthStatus {
  status: string;
  version: string;
  model_loaded: boolean;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Custom Waveform Canvas Renderer using Web Audio API
const WaveformCanvas: React.FC<{ file: File | null }> = ({ file }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);

  useEffect(() => {
    if (!file) {
      setAudioBuffer(null);
      return;
    }

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const arrayBuffer = e.target?.result as ArrayBuffer;
        if (arrayBuffer) {
          const buffer = await audioCtx.decodeAudioData(arrayBuffer);
          setAudioBuffer(buffer);
        }
      } catch (err) {
        console.error("Error decoding audio data for waveform:", err);
      }
    };

    reader.readAsArrayBuffer(file);

    return () => {
      audioCtx.close();
    };
  }, [file]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!audioBuffer) {
      // Draw flatline
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.strokeStyle = "hsla(217, 30%, 30%, 0.5)";
      ctx.lineWidth = 2;
      ctx.stroke();
      return;
    }

    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / canvas.width);
    const amp = canvas.height / 2;

    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);

    // Create a sleek blue-purple gradient for the waveform
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, "hsl(217, 91%, 60%)");
    gradient.addColorStop(0.5, "hsl(270, 80%, 60%)");
    gradient.addColorStop(1, "hsl(346, 84%, 55%)");

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;

    for (let i = 0; i < canvas.width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[i * step + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }

    ctx.stroke();
  }, [audioBuffer]);

  return (
    <div style={{ marginTop: "10px", width: "100%" }}>
      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>
        {audioBuffer ? "Decoded Audio Waveform (16 kHz downmixed)" : "No audio loaded"}
      </p>
      <div style={{ background: "hsl(222, 20%, 8%)", borderRadius: "8px", padding: "8px" }}>
        <canvas ref={canvasRef} width={600} height={100} style={{ width: "100%", height: "80px", display: "block" }} />
      </div>
    </div>
  );
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [history, setHistory] = useState<PredictionResult[]>([]);

  // Health and model checking on mount
  useEffect(() => {
    checkHealth();
    fetchModels();

    // Check health every 10 seconds
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: "error", version: "0.1.0", model_loaded: false });
      }
    } catch (err) {
      setHealth({ status: "offline", version: "0.1.0", model_loaded: false });
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || []);
      }
    } catch (err) {
      console.error("Failed to fetch models list:", err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setupFile(e.target.files[0]);
    }
  };

  const setupFile = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(URL.createObjectURL(selectedFile));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setupFile(e.dataTransfer.files[0]);
    }
  };

  const submitFile = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Server failed to process audio");
      }

      const data = await response.json();
      const newResult: PredictionResult = {
        prediction: data.prediction,
        confidence: data.confidence,
        score: data.score,
        timestamp: data.timestamp,
        model_name: data.model_name || "acoustic_detector",
        feature_importance: data.feature_importance,
        explanation: data.explanation,
        filename: file.name,
        latency_ms: data.latency_ms,
      };

      setResult(newResult);
      setHistory(prev => [newResult, ...prev]);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  // Convert markdown-like explanations to standard paragraphs
  const formatExplanation = (text: string) => {
    if (!text) return null;
    return text.split("\n").map((line, idx) => {
      if (line.startsWith("- **")) {
        // Bullet points with bold text
        const match = line.match(/^-\s+\*\*(.*?)\*\*:\s*(.*)/);
        if (match) {
          return (
            <li key={idx} style={{ marginLeft: "20px", marginBottom: "8px" }}>
              <strong>{match[1]}:</strong> {match[2]}
            </li>
          );
        }
      }
      if (line.startsWith("Classified as")) {
        return <p key={idx} style={{ fontWeight: "600", color: "var(--text-primary)", marginBottom: "12px" }}>{line}</p>;
      }
      return <p key={idx} style={{ marginBottom: "8px" }}>{line}</p>;
    });
  };

  // Group feature importance scores into meaningful bins
  const getFeatureImportanceRows = (importance: number[]) => {
    if (!importance || importance.length === 0) return [];
    
    // We group the 64 Mel channels into 5 frequency sub-bands:
    // Sub-bass (bins 0-5), Bass (bins 6-15), Midrange (bins 16-35), High Mid (bins 36-50), Highs (bins 51-63)
    const bands = [
      { label: "Sub-Bass", start: 0, end: 5 },
      { label: "Bass", start: 6, end: 15 },
      { label: "Midrange", start: 16, end: 35 },
      { label: "High-Mids", start: 36, end: 50 },
      { label: "Highs", start: 51, end: 63 }
    ];

    return bands.map(band => {
      const slice = importance.slice(band.start, band.end + 1);
      const avg = slice.reduce((sum, v) => sum + Math.abs(v), 0) / slice.length;
      return { label: band.label, val: avg };
    });
  };

  // Find max value to normalize widths in bar chart
  const bandRows = result?.feature_importance ? getFeatureImportanceRows(result.feature_importance) : [];
  const maxVal = bandRows.length > 0 ? Math.max(...bandRows.map(r => r.val)) : 1;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header-container">
        <div className="title-group">
          <h1>
            <span style={{ color: "var(--color-primary)" }}>Acoustic</span>Space
          </h1>
          <p>Deepfake Audio Forensic & Environmental Acoustic Auditor</p>
        </div>
        <div className="health-badge">
          <span className={`dot ${health?.status === "healthy" ? "active" : "inactive"}`}></span>
          <span>
            API Status: {health?.status === "healthy" ? "HEALTHY" : health?.status || "CONNECTING..."}
          </span>
          {health?.version && <span style={{ color: "var(--text-muted)" }}>v{health.version}</span>}
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Sidebar Controls */}
        <section className="sidebar">
          {/* File Upload card */}
          <div className="glass-panel">
            <h2 style={{ fontSize: "1.2rem", marginBottom: "15px" }}>Audio File Ingression</h2>
            <div 
              className={`upload-zone ${isDragOver ? "dragover" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="upload-icon">⏏</div>
              <div>
                <p style={{ fontWeight: "500", color: "var(--text-primary)" }}>Drag & Drop Audio File</p>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
                  Supports FLAC, WAV, MP3 up to 50 MB
                </p>
              </div>
              <button 
                type="button" 
                style={{
                  background: "var(--color-primary)",
                  color: "white",
                  border: "none",
                  padding: "8px 18px",
                  borderRadius: "var(--radius-sm)",
                  cursor: "pointer",
                  fontWeight: "500",
                  marginTop: "5px",
                }}
                onClick={() => document.getElementById("file-upload")?.click()}
              >
                Browse Files
              </button>
              <input 
                id="file-upload" 
                type="file" 
                className="file-input" 
                accept=".flac,.wav,.mp3,.ogg,.m4a"
                onChange={handleFileChange}
              />
            </div>

            {file && (
              <div style={{ marginTop: "20px", display: "flex", flexDirection: "column", gap: "10px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.9rem", fontWeight: "600", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", maxWidth: "200px" }}>
                    {file.name}
                  </span>
                  <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>

                <WaveformCanvas file={file} />

                {audioUrl && (
                  <div className="audio-player-container">
                    <audio src={audioUrl} controls />
                  </div>
                )}

                <button
                  type="button"
                  disabled={loading}
                  style={{
                    width: "100%",
                    background: "linear-gradient(135deg, var(--color-primary), hsl(270, 80%, 60%))",
                    color: "white",
                    border: "none",
                    padding: "12px",
                    borderRadius: "var(--radius-md)",
                    cursor: loading ? "not-allowed" : "pointer",
                    fontWeight: "600",
                    fontSize: "0.95rem",
                    boxShadow: "0 4px 15px -4px hsla(217, 100%, 50%, 0.3)",
                    marginTop: "10px",
                    transition: "var(--transition)"
                  }}
                  onClick={submitFile}
                >
                  {loading ? "Extracting Acoustic Fingerprints..." : "Analyze Recording"}
                </button>
              </div>
            )}
            {error && (
              <div style={{ marginTop: "15px", padding: "10px", background: "hsla(346, 84%, 55%, 0.15)", border: "1px solid var(--color-fake)", borderRadius: "var(--radius-sm)", color: "var(--color-fake)", fontSize: "0.85rem" }}>
                <strong>Error: </strong> {error}
              </div>
            )}
          </div>

          {/* Model Status Card */}
          <div className="glass-panel">
            <h2 style={{ fontSize: "1.2rem", marginBottom: "15px" }}>Available Forensic Models</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {models.length > 0 ? (
                models.map((m, idx) => (
                  <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", background: "hsla(217, 20%, 12%, 0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)" }}>
                    <div style={{ display: "flex", flexDirection: "column" }}>
                      <span style={{ fontSize: "0.85rem", fontWeight: "600", textTransform: "capitalize" }}>
                        {m.name.replace("_", " ")}
                      </span>
                      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                        {m.type === "traditional_ml" ? "Classical ML" : "Deep Learning / Transformer"}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "var(--color-real)", background: "var(--color-real-glow)", padding: "2px 8px", borderRadius: "4px" }}>
                      {m.status}
                    </span>
                  </div>
                ))
              ) : (
                <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>No forensic model index loaded.</p>
              )}
            </div>
          </div>
        </section>

        {/* Prediction Results Panel */}
        <section style={{ display: "flex", flexDirection: "column", gap: "30px" }}>
          {loading && (
            <div className="glass-panel loading-overlay">
              <div className="spinner"></div>
              <p style={{ fontSize: "1.1rem", fontWeight: "500", color: "var(--text-primary)" }}>
                Auditing Audio acoustics & respiratory channels...
              </p>
              <div className="progress-container">
                <div className="progress-bar"></div>
              </div>
            </div>
          )}

          {!loading && !result && (
            <div className="glass-panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "80px 20px", textAlign: "center" }}>
              <span style={{ fontSize: "4rem", filter: "grayscale(100%)" }}>🎙️</span>
              <h2 style={{ color: "var(--text-primary)", fontSize: "1.3rem", marginTop: "15px" }}>Acoustic Fingerprint Auditor</h2>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", maxWidth: "400px", marginTop: "8px" }}>
                Ingress an audio sample to verify room reverberation characteristics, breathing dynamics, and vocoder anomalies.
              </p>
            </div>
          )}

          {!loading && result && (
            <div className="glass-panel result-card animate-fade-in">
              <div className="result-header">
                <div>
                  <h2 style={{ fontSize: "1.4rem" }}>Forensic Verdict</h2>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    Analyzed in {result.latency_ms ? `${result.latency_ms.toFixed(1)} ms` : "N/A"} via {result.model_name}
                  </p>
                </div>
                <span className={`verdict-tag ${result.prediction.toLowerCase()}`}>
                  {result.prediction}
                </span>
              </div>

              {/* Confidence Meter */}
              <div className="meter-container">
                <div className="meter-labels">
                  <span>Confidence Gauge</span>
                  <span style={{ fontWeight: "700", color: result.prediction.toLowerCase() === "real" ? "var(--color-real)" : "var(--color-fake)" }}>
                    {(result.confidence * 100).toFixed(1)}% Probability
                  </span>
                </div>
                <div className="meter-track">
                  <div 
                    className={`meter-bar ${result.prediction.toLowerCase()}`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Speech Explanation Card */}
              {result.explanation && (
                <div>
                  <h3 style={{ fontSize: "1rem", color: "var(--text-primary)", marginBottom: "8px" }}>White-Box RIR & Cadence Explanation</h3>
                  <div className="explanation-box">
                    {formatExplanation(result.explanation)}
                  </div>
                </div>
              )}

              {/* Custom styled Feature Importance bar chart */}
              {result.feature_importance && result.feature_importance.length > 0 && (
                <div className="chart-container">
                  <h3 className="chart-title">Acoustic Spectral Attributions</h3>
                  <div className="bar-grid">
                    {bandRows.map((row, idx) => (
                      <div className="bar-row" key={idx}>
                        <span className="bar-label">{row.label}</span>
                        <div className="bar-track">
                          <div 
                            className="bar-fill" 
                            style={{ 
                              width: `${(row.val / maxVal) * 100}%`,
                              background: result.prediction.toLowerCase() === "real" ? "var(--color-real)" : "var(--color-fake)"
                            }}
                          />
                        </div>
                        <span className="bar-val">{row.val.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* History Panel */}
          {history.length > 0 && (
            <div className="glass-panel">
              <h2 style={{ fontSize: "1.2rem", marginBottom: "15px" }}>Historical Analysis Logs</h2>
              <div className="history-panel">
                <div className="history-list">
                  {history.map((h, idx) => (
                    <div className="history-item" key={idx}>
                      <div className="history-info">
                        <span className="history-filename">{h.filename}</span>
                        <span className="history-time">
                          {new Date(h.timestamp).toLocaleTimeString()} - {h.model_name}
                        </span>
                      </div>
                      <span className={`history-badge ${h.prediction.toLowerCase()}`}>
                        {h.prediction} ({Math.round(h.confidence * 100)}%)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
