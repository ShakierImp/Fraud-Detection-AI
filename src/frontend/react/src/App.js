import React, { useState, useCallback } from "react";
import axios from "axios";
import "./App.css";
import UploadArea from "./components/UploadArea";
import ResultsTable from "./components/ResultsTable";
import MetricCard from "./components/MetricCard";

/**
 * Main app component - orchestrates upload, analysis and results.
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null); // API response: {predictions: [...], summary: {...}}
  const [error, setError] = useState("");

  // File selected callback from UploadArea
  const onFileSelected = useCallback((fileObj) => {
    setFile(fileObj);
    setFilename(fileObj ? fileObj.name : "");
    setResults(null);
    setError("");
  }, []);

  // Trigger analysis - POST file to backend
  const handleAnalyze = useCallback(async () => {
    if (!file) {
      setError("Please select a CSV file to analyze.");
      return;
    }
    setError("");
    setIsLoading(true);
    setResults(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const resp = await axios.post("http://localhost:8000/predict", form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000
      });
      setResults(resp.data);
    } catch (err) {
      console.error(err);
      setError(
        "Upload failed. Please ensure you've selected a CSV file and the backend (http://localhost:8000) is running."
      );
    } finally {
      setIsLoading(false);
    }
  }, [file]);

  // Compute metrics for MetricCards
  const total = results?.predictions?.length ?? 0;
  const flagged = results
    ? results.predictions.filter((p) => p.risk_score === "High" || p.risk_score === "Medium").length
    : 0;
  const flaggedPct = total > 0 ? ((flagged / total) * 100).toFixed(1) : "0.0";

  return (
    <div className="container">
      <div className="header">
        <div className="brand-logo">FG</div>
        <div>
          <h1>FraudGuardian AI — Suspicious Transactions Detector</h1>
          <p>Upload transactions (CSV) and detect suspicious activity using our AI engine.</p>
        </div>
      </div>

      <div className="top-row">
        <div>
          <UploadArea onFileChange={onFileSelected} selectedFileName={filename} />

          <div style={{ marginTop: 18, display: "flex", gap: 12 }}>
            <button
              className={`btn primary ${isLoading ? "loading" : ""}`}
              onClick={handleAnalyze}
              disabled={isLoading || !file}
            >
              {isLoading ? <span className="spinner" /> : "Analyze Transactions"}
            </button>

            <button
              className="btn ghost"
              onClick={() => {
                setFile(null);
                setFilename("");
                setResults(null);
                setError("");
              }}
            >
              Reset
            </button>
          </div>

          {error && <div className="error">{error}</div>}
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>Quick Summary</h3>
              <p className="muted">Status of uploaded dataset</p>
            </div>
          </div>

          <div className="metrics-row">
            <MetricCard title="Total Transactions" value={total} />
            <MetricCard title="Flagged Transactions" value={flagged} />
            <MetricCard title="Flagged %" value={`${flaggedPct}%`} />
          </div>

          <div style={{ marginTop: 12 }}>
            <small className="muted">Model status: <strong>Connected</strong></small>
          </div>
        </div>
      </div>

      {/* Results area appears only when results exist */}
      {results && (
        <section style={{ marginTop: 28 }}>
          <ResultsTable results={results} />
        </section>
      )}

      <footer style={{ marginTop: 40, color: "var(--muted)", fontSize: 13 }}>
        Built with ❤️ for the hackathon — FraudGuardian AI. Backend: <code>http://localhost:8000</code>
      </footer>
    </div>
  );
}
