import React, { useMemo } from "react";
import "./ResultsTable.css";

/**
 * ResultsTable
 * Props:
 *  - results: { predictions: [{transaction_id, probability, risk_score}], summary: {...} }
 */
export default function ResultsTable({ results }) {
  const predictions = results?.predictions || [];

  // Prepare flagged CSV download content
  const flaggedRows = useMemo(
    () => predictions.filter((p) => p.risk_score === "High" || p.risk_score === "Medium"),
    [predictions]
  );

  const downloadFlagged = () => {
    if (!flaggedRows.length) return;
    const header = Object.keys(flaggedRows[0]).join(",");
    const rows = flaggedRows.map((r) =>
      [r.transaction_id, r.probability, r.risk_score].join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "flagged_transactions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="results-panel">
      <div className="results-header">
        <div>
          <h2>Analysis Results</h2>
          <p className="muted">Found {predictions.length} transactions · Flagged: {results.summary.High + results.summary.Medium}</p>
        </div>
        <div className="results-actions">
          <button className="btn ghost" onClick={downloadFlagged} disabled={!flaggedRows.length}>
            Download Flagged CSV
          </button>
        </div>
      </div>

      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Probability</th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p, i) => (
              <tr key={`${p.transaction_id}-${i}`} className={i % 2 === 0 ? "even" : "odd"}>
                <td>{p.transaction_id}</td>
                <td>{(p.probability * 100).toFixed(2)}%</td>
                <td>
                  <span className={`badge ${p.risk_score.toLowerCase()}`}>
                    {p.risk_score}
                  </span>
                </td>
              </tr>
            ))}
            {!predictions.length && (
              <tr>
                <td colSpan="3">
                  <div className="empty">No results to display.</div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
