import React, { useRef, useState, useCallback } from "react";
import "./UploadArea.css";

/**
 * UploadArea
 * - Drag-and-drop and click-to-browse area for CSV uploads
 * - Accepts single .csv file
 */
export default function UploadArea({ onFileChange, selectedFileName }) {
  const fileRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [localName, setLocalName] = useState(selectedFileName || "");

  const preventDefaults = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    preventDefaults(e);
    setDragging(false);
    const dt = e.dataTransfer;
    const file = dt?.files?.[0];
    if (file && file.name.endsWith(".csv")) {
      setLocalName(file.name);
      onFileChange(file);
    } else {
      alert("Please upload a CSV file.");
    }
  };

  const handleFile = (file) => {
    if (file && file.name.endsWith(".csv")) {
      setLocalName(file.name);
      onFileChange(file);
    } else {
      alert("Only CSV files are accepted.");
    }
  };

  const onBrowseClick = () => {
    fileRef.current?.click();
  };

  const onFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
  };

  const onDragOver = (e) => {
    preventDefaults(e);
    setDragging(true);
  };

  const onDragLeave = (e) => {
    preventDefaults(e);
    setDragging(false);
  };

  return (
    <div className="upload-wrapper">
      <div
        className={`upload-area ${dragging ? "dragging" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={handleDrop}
        onClick={onBrowseClick}
        role="button"
        tabIndex={0}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={onFileInputChange}
        />
        <div className="upload-inner">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
            <path d="M12 3v10" stroke="#2b6cb0" strokeWidth="1.6" strokeLinecap="round"/>
            <path d="M5 10l7-7 7 7" stroke="#2b6cb0" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M21 21H3" stroke="#cbd5e1" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
          <div>
            <div className="title">Drag & drop CSV here, or click to browse</div>
            <div className="muted">Only .csv files accepted. Max single file.</div>
            {localName && <div className="filename">Selected: {localName}</div>}
          </div>
        </div>
      </div>
      <div className="hint">Tip: Include a <code>transaction_id</code> column to map results back.</div>
    </div>
  );
}

