import React from "react";
import "./MetricCard.css";

/**
 * MetricCard - small stat display used in the top-right panel
 */
export default function MetricCard({ title, value }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-title">{title}</div>
    </div>
  );
}
