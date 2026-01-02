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

            data.forEach(row => {
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
