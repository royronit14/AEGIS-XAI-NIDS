const API_BASE = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------
    // HEALTH CHECK
    // -----------------------------
    fetch(`${API_BASE}/health`)
        .then(res => res.json())
        .then(data => {
            console.log("Health response:", data);
            const el = document.getElementById("health-status");
            el.innerText =
                data.status === "ok"
                    ? "🟢 System Operational"
                    : "🔴 System Down";
        })
        .catch(err => {
            console.error("Health check failed:", err);
            document.getElementById("health-status").innerText =
                "🔴 Backend not reachable";
        });
});

// -----------------------------
// LOCAL EXPLANATION
// -----------------------------
function getExplanation() {
    event?.preventDefault?.();
    const index = document.getElementById("alertIndex").value;

    fetch(`${API_BASE}/explain/local`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            dataset: "cicids",
            index: Number(index)
        })
    })
        .then(res => res.json())
        .then(data => {
            document.getElementById("explain-output").innerText =
                JSON.stringify(data, null, 2);
        });
}

// -----------------------------
// DRIFT MONITOR
// -----------------------------
function getDrift() {
    fetch(`${API_BASE}/drift/features`)
        .then(res => res.json())
        .then(data => {
            const tbody = document.querySelector("#drift-table tbody");
            tbody.innerHTML = "";

            data.drift_metrics.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${row.feature}</td>
                    <td>${row.psi}</td>
                    <td class="status-${row.status}">${row.status}</td>
                `;
                tbody.appendChild(tr);
            });
        });
}

// =============================
// STEP-6: DRIFT HISTORY
// =============================
function loadDriftHistory() {
    fetch(`${API_BASE}/history/drift`)
        .then(res => res.json())
        .then(data => {
            const x = data.map(d => d.timestamp);
            const y = data.map(d => d.psi);

            Plotly.newPlot("drift-history-chart", [{
                x,
                y,
                mode: "lines+markers",
                name: "PSI"
            }], {
                title: "Drift Trend Over Time",
                yaxis: { title: "PSI Score" }
            });
        })
        .catch(err => console.error("Drift history error:", err));
}

// =============================
// STEP-6: PERFORMANCE HISTORY
// =============================
function loadPerformanceHistory() {
    fetch(`${API_BASE}/history/performance`)
        .then(res => res.json())
        .then(data => {
            Plotly.newPlot("performance-history-chart", [
                {
                    x: data.map(d => d.timestamp),
                    y: data.map(d => d.accuracy),
                    name: "Accuracy",
                    mode: "lines"
                },
                {
                    x: data.map(d => d.timestamp),
                    y: data.map(d => d.f1),
                    name: "F1 Score",
                    mode: "lines"
                }
            ], {
                title: "Model Performance Over Time"
            });
        })
        .catch(err => console.error("Performance history error:", err));
}

// =============================
// STEP-6: RETRAINING HISTORY
// =============================
function loadRetrainingHistory() {
    fetch(`${API_BASE}/history/retraining`)
        .then(res => res.json())
        .then(data => {
            Plotly.newPlot("retraining-history-chart", [{
                x: data.map(d => d.timestamp),
                y: data.map(() => 1),
                mode: "markers",
                marker: { size: 14 },
                text: data.map(d => d.reason),
                hoverinfo: "text"
            }], {
                title: "Retraining & Policy Actions",
                yaxis: { visible: false }
            });
        })
        .catch(err => console.error("Retraining history error:", err));
}
