import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [latexResult, setLatexResult] = useState("");

  const fileInputRef = useRef(null);

  const handleBrowseClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) return;
    setImageFile(file);
    setLatexResult("");
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleFileChange = (e) => {
    handleFiles(e.target.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleConvert = async () => {
    if (!imageFile || isConverting) return;

    setIsConverting(true);
    setLatexResult("");
    setProgress(0);

    // Fake progress bar + placeholder backend call
    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + 7;
        if (next >= 100) {
          clearInterval(interval);
          setIsConverting(false);

          // TODO: replace this with real API call
          // e.g. const result = await fetch("/api/convert", ...)
          setLatexResult("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}");

          return 100;
        }
        return next;
      });
    }, 80);
  };

  const handleCopy = async () => {
    if (!latexResult) return;
    try {
      await navigator.clipboard.writeText(latexResult);
      // optional: brief visual feedback
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header py-8">
        <h1 className="app-title mb-5">Texify</h1>
        <p className="app-subtitle">
          Convert your handwritten mathematical expressions into LaTeX!
        </p>
      </header>

      <main className="app-main">
        {/* Upload box */}
        <section
            className={`upload-area ${isDragging ? "upload-area--dragging" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
          <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              className="upload-input"
              onChange={handleFileChange}
          />
          <button type="button" className="upload-button" onClick={handleBrowseClick}>
            Upload image
          </button>
          <p className="upload-hint">or drag &amp; drop an image here</p>
        </section>

        {/* Image + controls */}
        <section className="work-area">
          <div className="image-panel">
            <div className="image-box">
              {previewUrl ? (
                  <img
                      src={previewUrl}
                      alt="Uploaded preview"
                      className="preview-image"
                  />
              ) : (
                  <div className="image-placeholder">
                    Uploaded image will appear here
                  </div>
              )}
            </div>

            {!isConverting && !latexResult && (
              <div className="convert-center">
                <button
                    type="button"
                    className="convert-button"
                    onClick={handleConvert}
                    disabled={!imageFile}
                >
                  Convert!
                </button>
              </div>
            )}

            {/* Progress bar also centered under the button */}
            {isConverting && (
                <div className="progress-wrapper progress-center">
                  <div className="progress-label">Processing…</div>
                  <div className="progress-bar">
                    <div
                        className="progress-bar-fill"
                        style={{width: `${progress}%`}}
                    />
                  </div>
                </div>
            )}

            {/* Result output stays below everything */}
            {!isConverting && latexResult && (
                <div className="result-container result-center">
                  <div className="result-header">
                    <span>LaTeX output</span>
                    <button
                        type="button"
                        className="copy-button"
                        onClick={handleCopy}
                    >
                      copy
                    </button>
                  </div>
                  <pre className="result-box">{latexResult}</pre>
                </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;