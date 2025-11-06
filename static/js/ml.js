// JS to interact with ML endpoints and render results on the site map

let ML_SIGHTINGS = [];
let LAST_PREDICTION = null;

function updateSightingsList() {
    const el = document.getElementById('sightings-list');
    if (!el) return;
    if (ML_SIGHTINGS.length === 0) {
        el.innerHTML = '<p class="text-muted">No sightings added yet.</p>';
        return;
    }
    let html = '<ul class="list-group">';
    ML_SIGHTINGS.forEach((s, idx) => {
        html += `<li class="list-group-item d-flex justify-content-between align-items-center">${s.direction_text} (${s.hours_since}h) <span><button class="btn btn-sm btn-danger" onclick="removeSighting(${idx})">Remove</button></span></li>`;
    });
    html += '</ul>';
    el.innerHTML = html;
}

function removeSighting(idx) {
    ML_SIGHTINGS.splice(idx,1);
    updateSightingsList();
}

async function postPredict(payload) {
    const res = await fetch('/api/ml/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    return res.json();
}

async function postRefine(payload) {
    const res = await fetch('/api/ml/refine', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    return res.json();
}

function showPrediction(pred, input) {
    const container = document.getElementById('prediction-cards');
    if (!container) return;
    if (!pred) {
        container.innerHTML = '';
        return;
    }
    LAST_PREDICTION = pred;
    const risk_map = {0: 'Low', 1: 'Medium', 2: 'High'};
    const html = `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">Risk: ${risk_map[pred.risk_label] || pred.risk_label}</h5>
                <p>Risk Confidence: ${(pred.risk_prob*100).toFixed(1)}%</p>
                <p>Probability of Recovery: ${(pred.recovered_prob*100).toFixed(1)}%</p>
                <p>Estimated Recovery Time: ${Math.round(pred.recovery_time_hours)} hours</p>
            </div>
        </div>
    `;
    container.innerHTML = html;

    // Update map markers
    if (window.ml_map) {
        window.ml_map.eachLayer(function(layer){ if (layer && layer._url==null) window.ml_map.removeLayer(layer); });
        const lat = input.latitude || 18.5203;
        const lon = input.longitude || 73.8567;
        addMarkerToMap(window.ml_map, lat, lon, 'Last Seen', 'blue');
        const p_lat = pred.predicted_latitude || 0;
        const p_lon = pred.predicted_longitude || 0;
        addMarkerToMap(window.ml_map, p_lat, p_lon, 'Initial Hotspot', 'red');
        window.ml_map.setView([p_lat || lat, p_lon || lon], 10);
    }
}

function enableFormHandlers() {
    document.getElementById('predict-btn').addEventListener('click', async function(e){
        e.preventDefault();
        const form = document.getElementById('ml-form');
        const fd = new FormData(form);
        const payload = Object.fromEntries(fd.entries());
        // parse numeric fields
        payload.latitude = parseFloat(payload.latitude);
        payload.longitude = parseFloat(payload.longitude);
        payload.child_age = parseInt(payload.child_age);
        payload.abduction_time = parseInt(payload.abduction_time);
        payload.population_density = parseFloat(payload.population_density || 0);
        payload.transport_hub_nearby = payload.transport_hub_nearby ? 1 : 0;

        const res = await postPredict(payload);
        if (res.success) {
            showPrediction(res.prediction, res.case_input);
        } else {
            alert('Prediction failed: ' + (res.error || 'unknown'));
        }
    });

    document.getElementById('add-sighting-btn').addEventListener('click', function(e){
        const sf = document.getElementById('sighting-form');
        const fd = new FormData(sf);
        const s = Object.fromEntries(fd.entries());
        s.lat = parseFloat(s.s_lat);
        s.lon = parseFloat(s.s_lon);
        s.hours_since = parseFloat(s.s_hours);
        s.direction_text = s.s_text || '';
        ML_SIGHTINGS.push({lat: s.lat, lon: s.lon, hours_since: s.hours_since, direction_text: s.direction_text});
        updateSightingsList();
    });

    document.getElementById('refine-btn').addEventListener('click', async function(e){
        if (!LAST_PREDICTION) { alert('Run an initial prediction first'); return; }
        const payload = {
            initial_prediction: LAST_PREDICTION,
            sightings: ML_SIGHTINGS,
            initial_case_input: LAST_PREDICTION ? {} : {}
        };
        // try to include last case_input if available on page; the predict flow stores it in window via response
        // For simplicity, call predict again to regenerate case_input and use that
        // Here assume LAST_PREDICTION came from server that also returned case_input

        // If LAST_PREDICTION contains case_input
        if (LAST_PREDICTION.case_input) payload.initial_case_input = LAST_PREDICTION.case_input;

        const res = await postRefine(payload);
        if (res.success) {
            const rlat = res.refined_lat; const rlon = res.refined_lon;
            addMarkerToMap(window.ml_map, rlat, rlon, 'Refined Hotspot', 'purple');
            window.ml_map.setView([rlat, rlon], 11);
            showNotification('Refined location: ' + rlat.toFixed(4) + ', ' + rlon.toFixed(4), 'success');
        } else {
            alert('Refine failed: ' + (res.error || 'unknown'));
        }
    });
}

// Wire up handlers after DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enableFormHandlers);
} else {
    enableFormHandlers();
}
