import React, { useState, useRef } from "react";
import logoMain from "./assets/logo_main.png";
import "./App.css";

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [latexResult, setLatexResult] = useState("");
  const [copied, setCopied] = useState(false);

  const [usingWebcam, setUsingWebcam] = useState(false);
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleBrowseClick = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) return;

    setUsingWebcam(false);
    setImageFile(file);
    setLatexResult("");
    setPreviewUrl(URL.createObjectURL(file));
    setCopied(false);
  };

  const handleFileChange = (e) => handleFiles(e.target.files);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  // ------------------------------------
  // ⭐ NEW: Start webcam preview
  // ------------------------------------
  const handleUseWebcam = async () => {
    setUsingWebcam(true);
    setPreviewUrl(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      console.error("Webcam error:", err);
      alert("Unable to access webcam.");
      setUsingWebcam(false);
    }
  };

  // ------------------------------------
  // ⭐ NEW: Capture frame from webcam
  // ------------------------------------
  const handleTakePhoto = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) return;

      const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
      setLatexResult("");
      setCopied(false);
      setUsingWebcam(false);

      // stop webcam
      const stream = video.srcObject;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    }, "image/jpeg");
  };

  // Convert button (unchanged)
  const handleConvert = async () => {
    if (!imageFile) return;

    setIsConverting(true);
    setLatexResult("");
    setCopied(false);
    setProgress(0);

    const formData = new FormData();
    formData.append("file", imageFile);

    let current = 0;
    const progressTimer = setInterval(() => {
      current += Math.random() * 5 + 2;
      if (current < 90) setProgress(current);
    }, 150);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/convert", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      clearInterval(progressTimer);
      setProgress(100);

      setLatexResult(data.latex || "No LaTeX output.");
    } catch (err) {
      console.error(err);
      clearInterval(progressTimer);
      setProgress(100);
      setLatexResult("ERROR: backend unreachable or crashed.");
    }

    setIsConverting(false);
  };

  const handleCopy = async () => {
    if (!latexResult) return;
    await navigator.clipboard.writeText(latexResult);
    setCopied(true);
  };

  return (
    <div className="app-root">
      <header className="app-header py-8">
        <img src={logoMain} alt="Texify logo" className="app-logo" />
        <p className="app-subtitle">Convert your handwritten mathematical expressions into LaTeX!</p>
      </header>

      <main className="app-main">
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

          {/* Webcam OR Upload options */}
          {!usingWebcam && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <button className="upload-button" onClick={handleUseWebcam}>
                Take a pic
              </button>

              <div style={{ margin: "10px", fontWeight: 500, opacity: 0.7 }}>or</div>

              <button className="upload-button" onClick={handleBrowseClick}>
                Upload Image
              </button>
            </div>
          )}

          {/* ⭐ Webcam live preview + Cheese button */}
          {usingWebcam && (
              <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                <video
                    ref={videoRef}
                    autoPlay
                    style={{
                      width: "100%",
                      maxWidth: "380px",
                      borderRadius: "8px",
                      marginBottom: "12px",
                      marginTop: "10px"
                    }}
                />

                <button
                    type="button"
                    className="upload-button cheese-button"
                    onClick={handleTakePhoto}
                >
                  📸 Cheese!
                </button>
              </div>
          )}

          {!usingWebcam && (
              <p className="upload-hint">or drag & drop an image here</p>
          )}
        </section>

        {/* Image + convert displays */}
        <section className="work-area">
          <div className="image-panel">
            <div className="image-box">
              {previewUrl ? (
                <img src={previewUrl} className="preview-image" alt="Preview" />
              ) : (
                <div className="image-placeholder">Uploaded image will appear here</div>
              )}
            </div>

            {!isConverting && !latexResult && (
              <div className="convert-center">
                <button className="convert-button" onClick={handleConvert} disabled={!imageFile}>
                  Convert!
                </button>
              </div>
            )}

            {isConverting && (
              <div className="progress-wrapper progress-center">
                <div className="progress-label">Processing…</div>
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {!isConverting && latexResult && (
              <div className="result-container result-center">
                <div className="result-header">
                  <span>LaTeX output</span>
                  <button
                    className={`copy-button ${copied ? "copy-button-disabled" : ""}`}
                    onClick={handleCopy}
                    disabled={copied}
                  >
                    {copied ? "✔ copied" : "copy"}
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