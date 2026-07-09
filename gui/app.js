// Configuration metrics dictionary for bottom right dashboard
const PIPELINE_METRICS = {
    "quantum_a_svm": { accuracy: "89.84%", f1: "89.83%", pvalue: "0.0037" },
    "quantum_b_xgb": { accuracy: "89.29%", f1: "89.28%", pvalue: "0.0110" },
    "baseline_a_svm": { accuracy: "87.00%", f1: "86.99%", pvalue: "Ref" },
    "baseline_b_xgb": { accuracy: "87.00%", f1: "86.96%", pvalue: "Ref" }
};

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const modelSelector = document.getElementById("model-selector");
    const dropZone = document.getElementById("drop-zone");
    const fileSelectBtn = document.getElementById("file-select-btn");
    const fileInput = document.getElementById("file-input");
    const randomBtn = document.getElementById("random-btn");
    
    const inputMri = document.getElementById("input-mri");
    const preprocessedMri = document.getElementById("preprocessed-mri");
    const explainImage = document.getElementById("explain-image");
    
    const predictionResult = document.getElementById("prediction-result");
    const diagBox = document.getElementById("diag-box");
    const groundTruthLabel = document.getElementById("ground-truth-label");
    
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    
    // Metrics DOM
    const metricAccuracy = document.getElementById("metric-accuracy");
    const metricF1 = document.getElementById("metric-f1");
    const metricPvalue = document.getElementById("metric-pvalue");

    let currentBase64Image = null;

    // Initialize metrics based on default selection
    updateMetricsDisplay(modelSelector.value);

    // Event Listeners
    modelSelector.addEventListener("change", (e) => {
        const val = e.target.value;
        updateMetricsDisplay(val);
        // If an image is already loaded, re-predict with the new model immediately
        if (currentBase64Image) {
            runDiagnosis(currentBase64Image, val);
        }
    });

    // File Input Browse button
    fileSelectBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    // Drag & Drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    // Random Dataset scan loader
    randomBtn.addEventListener("click", () => {
        showLoading("Fetching Random Dataset Scan...", "Browsing through testing subset folders...");
        fetch("/api/random_test")
            .then(res => {
                if (!res.ok) throw new Error("Could not fetch random scan.");
                return res.json();
            })
            .then(data => {
                if (data.success) {
                    currentBase64Image = data.image;
                    inputMri.src = data.image;
                    
                    // Show ground truth label
                    groundTruthLabel.style.display = "block";
                    groundTruthLabel.innerHTML = `<i class="fa-solid fa-tag"></i> Ground Truth: <strong>${data.label}</strong>`;
                    
                    // Run diagnosis
                    runDiagnosis(data.image, modelSelector.value);
                } else {
                    alert("Error loading scan from dataset.");
                    hideLoading();
                }
            })
            .catch(err => {
                console.error(err);
                alert("Failed to load random scan from dataset. Make sure server is running.");
                hideLoading();
            });
    });

    // Helper functions
    function updateMetricsDisplay(modelVal) {
        const met = PIPELINE_METRICS[modelVal];
        if (met) {
            metricAccuracy.innerText = met.accuracy;
            metricF1.innerText = met.f1;
            metricPvalue.innerText = met.pvalue;
        }
    }

    function handleFile(file) {
        if (!file.type.match("image.*")) {
            alert("Please upload a valid image file.");
            return;
        }
        
        // Hide ground truth as this is a custom upload
        groundTruthLabel.style.display = "none";
        
        const reader = new FileReader();
        reader.onload = (e) => {
            currentBase64Image = e.target.result;
            inputMri.src = currentBase64Image;
            runDiagnosis(currentBase64Image, modelSelector.value);
        };
        reader.readAsDataURL(file);
    }

    function runDiagnosis(base64Image, modelName) {
        showLoading("Running Medical Diagnosis Pipeline...", "Executing Preprocessing & Feature Fusion...");
        
        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: base64Image,
                model: modelName
            })
        })
        .then(res => {
            if (!res.ok) throw new Error("Inference failed.");
            return res.json();
        })
        .then(data => {
            if (data.success) {
                // Update images (with cache buster timestamps to avoid caching issues)
                preprocessedMri.src = data.preprocessed_url + "?t=" + new Date().getTime();
                explainImage.src = data.explainability_url + "?t=" + new Date().getTime();
                
                // Update diagnostic results
                predictionResult.innerText = data.prediction;
                
                // Reset classes and apply correct styling
                predictionResult.className = "diag-value";
                predictionResult.classList.add(data.prediction.toLowerCase().replace(/\s+/g, ''));
                
                // Update progress bars
                Object.keys(data.confidence).forEach(key => {
                    const probVal = data.confidence[key];
                    const pct = Math.round(probVal * 100);
                    
                    const pctEl = document.getElementById(`prob-${key}-pct`);
                    const fillEl = document.getElementById(`prob-${key}-fill`);
                    
                    if (pctEl && fillEl) {
                        pctEl.innerText = `${pct}%`;
                        fillEl.style.width = `${pct}%`;
                    }
                });
                
                // Render diagnostic explanation
                if (data.explanation) {
                    const explainSection = document.getElementById("explanation-section");
                    const explainContent = document.getElementById("explanation-content");
                    
                    // Parse explanation text into styled HTML
                    const lines = data.explanation.split("\n");
                    let html = "";
                    for (const line of lines) {
                        if (line.startsWith("DIAGNOSIS:")) {
                            html += `<div class="explain-diagnosis">${line}</div>`;
                        } else if (line.startsWith("Confidence:")) {
                            html += `<div class="explain-confidence">${line}</div>`;
                        } else if (line.endsWith(":") || line.startsWith("CLINICAL") || line.startsWith("FEATURE") || line.startsWith("SPATIAL") || line.startsWith("DIFFERENTIAL")) {
                            html += `<div class="explain-heading">${line}</div>`;
                        } else if (line.startsWith("•")) {
                            const isWarning = line.includes("Note:") || line.includes("narrow");
                            html += `<div class="explain-bullet ${isWarning ? 'explain-warning' : ''}">${line}</div>`;
                        } else if (line.trim() === "") {
                            html += `<br/>`;
                        } else {
                            html += `<div>${line}</div>`;
                        }
                    }
                    
                    explainContent.innerHTML = html;
                    explainSection.style.display = "block";
                    // Re-trigger animation
                    explainSection.style.animation = "none";
                    explainSection.offsetHeight; // force reflow
                    explainSection.style.animation = "slideIn 0.4s ease";
                }
                
            } else {
                alert("Diagnostic pipeline error: " + data.error);
            }
            hideLoading();
        })
        .catch(err => {
            console.error(err);
            alert("Connection to diagnostic server lost. Check terminal server output.");
            hideLoading();
        });
    }

    function showLoading(title, subtitle) {
        loadingText.innerText = title;
        document.querySelector(".loading-subtext").innerText = subtitle;
        loadingOverlay.classList.add("active");
    }

    function hideLoading() {
        loadingOverlay.classList.remove("active");
    }
});
