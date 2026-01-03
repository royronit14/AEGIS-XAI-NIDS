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
function getDrift(simulate = false) {
    fetch(`${API_BASE}/drift/features?simulate=${simulate}`)
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

            // 🔔 Update Alert Index Gauge
            updateAlertGauge(data.alert_index, data.drift_level);
        });
}


// =============================
// STEP-6: DRIFT HISTORY
// =============================
function loadDriftHistory() {
    fetch(`${API_BASE}/history/drift`)
        .then(res => res.json())
        .then(res => {
            console.log("Drift history:", res); // ✅ correct place

            const data = res.data;
            if (!data || !data.length) return;

            Plotly.newPlot("drift-history-chart", [{
                x: data.map(d => d.timestamp),
                y: data.map(d => d.psi),
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
        .then(res => {
            const data = res.data;
            if (!data || !data.length) return;

            Plotly.newPlot("performance-history-chart", [{
                x: data.map(d => d.timestamp),
                y: data.map(d => d.status === "healthy" ? 1 : 0),
                name: "Health Status",
                mode: "lines+markers"
            }], {
                title: "Model Health Over Time",
                yaxis: {
                    tickvals: [0, 1],
                    ticktext: ["Stale", "Healthy"]
                }
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
        .then(res => {
            const data = res.data;
            if (!data || !data.length) return;

            Plotly.newPlot("retraining-history-chart", [{
                x: data.map(d => d.created_at),
                y: data.map(() => 1),
                mode: "markers",
                marker: { size: 14 },
                text: data.map(d => d.reason || "Retrain Scheduled"),
                hoverinfo: "text"
            }], {
                title: "Retraining & Policy Actions",
                yaxis: { visible: false }
            });
        })
        .catch(err => console.error("Retraining history error:", err));
}

function updateAlertGauge(alertIndex, driftLevel) {
    const color =
        alertIndex < 0.3 ? "green" :
        alertIndex < 0.6 ? "orange" :
        "red";

    const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: alertIndex,
        title: { text: "System Risk Level" },
        gauge: {
            axis: { range: [0, 1] },
            bar: { color: color },
            steps: [
                { range: [0, 0.3], color: "#c8f7c5" },
                { range: [0.3, 0.6], color: "#ffeaa7" },
                { range: [0.6, 1], color: "#fab1a0" }
            ],
            threshold: {
                line: { color: "black", width: 4 },
                thickness: 0.75,
                value: alertIndex
            }
        }
    }];

    const layout = {
        height: 300,
        margin: { t: 40, b: 0 },
    };

    Plotly.newPlot("alert-gauge", data, layout);
}
