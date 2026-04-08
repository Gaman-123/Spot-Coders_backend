const API_BASE = 'http://localhost:8000';

// ── Element refs ─────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const runBtn         = document.getElementById('run-btn');
const consoleOutput  = document.getElementById('console-output');
const resultsContent = document.getElementById('results-content');
const runIdDisplay   = document.getElementById('run-id-display');
const backendStatus  = document.getElementById('backend-status');
const pipelineStatus = document.getElementById('pipeline-status');

// ── Backend health check ─────────────────────────────────────────────────────
async function checkBackend() {
    try {
        const res  = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'ok') {
            backendStatus.textContent    = 'ONLINE';
            backendStatus.style.color   = '#10b981';
        } else {
            backendStatus.textContent    = 'DEGRADED';
            backendStatus.style.color   = '#f59e0b';
        }
    } catch {
        backendStatus.textContent  = 'OFFLINE';
        backendStatus.style.color = '#ef4444';
    }
}
setInterval(checkBackend, 5000);
checkBackend();

// ── CSV Drop zone ─────────────────────────────────────────────────────────────
// FIX: was missing proper event binding — onclick + drag events now all wired
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files.length > 0) {
        // Create a new DataTransfer to assign files to the hidden input
        const dt = new DataTransfer();
        dt.items.add(e.dataTransfer.files[0]);
        fileInput.files = dt.files;
        updateDropZoneText();
    }
});

fileInput.addEventListener('change', updateDropZoneText);

function updateDropZoneText() {
    const dropText = document.getElementById('drop-text');
    if (fileInput.files && fileInput.files.length > 0) {
        dropText.textContent = `✓ ${fileInput.files[0].name}`;
        dropZone.style.borderColor = '#10b981';
        dropZone.style.background  = 'rgba(16, 185, 129, 0.08)';
    } else {
        dropText.textContent = 'Click or drag CSV/XLSX here';
        dropZone.style.borderColor = '';
        dropZone.style.background  = '';
    }
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
const modeRadios        = document.getElementsByName('run-mode');
const extraFilesSection = document.getElementById('extra-files-section');
const targetColumnGroup = document.querySelector('.form-group');

function updateModeVisibility() {
    const mode = Array.from(modeRadios).find(r => r.checked)?.value || 'upload';
    if (mode === 'pipeline') {
        dropZone.style.display         = 'none';
        extraFilesSection.style.display = 'none';
        targetColumnGroup.style.display = 'none';
        logToConsole('Pipeline mode: system will fetch latest dataset from DE automatically.', 'system');
    } else {
        dropZone.style.display         = 'block';
        extraFilesSection.style.display = 'flex';
        targetColumnGroup.style.display = 'block';
    }
}
modeRadios.forEach(r => r.addEventListener('change', updateModeVisibility));
updateModeVisibility();

// ── Console logger ─────────────────────────────────────────────────────────────
function logToConsole(message, type = 'system') {
    const line       = document.createElement('div');
    line.className   = `console-line ${type}`;
    const timestamp  = new Date().toLocaleTimeString([], { hour12: false });
    line.textContent = `[${timestamp}] ${message}`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// ── SSE stream ────────────────────────────────────────────────────────────────
function connectSSE(runId) {
    logToConsole(`Connecting to live data stream for run ${runId}...`, 'system');
    const eventSource = new EventSource(`${API_BASE}/run/${runId}/stream`);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'agent_update') {
                const agentName = (data.agent || '').replace('zora_', '').toUpperCase();
                const badge     = data.status === 'completed' ? '✓' :
                                  data.status === 'failed'    ? '✗' : '⟳';
                logToConsole(`${badge} [${agentName}] ${data.output_summary}`, data.status === 'failed' ? 'error' : 'agent');
                pipelineStatus.textContent = agentName;

            } else if (data.type === 'error') {
                logToConsole(`✗ ERROR: ${data.error_message}`, 'error');
                pipelineStatus.textContent = 'FAILED';
                eventSource.close();
                runBtn.disabled   = false;
                runBtn.textContent = 'Initialize Pipeline';

            } else if (data.type === 'pipeline_complete') {
                logToConsole('✓ PIPELINE COMPLETE — all agents finished.', 'success');
                pipelineStatus.textContent = 'COMPLETED';
                eventSource.close();
                fetchResults(runId);
                runBtn.disabled   = false;
                runBtn.textContent = 'Initialize Pipeline';
            }
        } catch (err) {
            console.error('SSE parse error:', err);
        }
    };

    eventSource.onerror = () => {
        logToConsole('Stream disconnected (pipeline may still be running in backend).', 'system');
        eventSource.close();
        runBtn.disabled   = false;
        runBtn.textContent = 'Initialize Pipeline';
    };
}

// ── Fetch results ─────────────────────────────────────────────────────────────
// FIX: was declared without `async function` wrapper — broke entire script
async function fetchResults(runId) {
    try {
        const res  = await fetch(`${API_BASE}/api/run/${runId}/status`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        resultsContent.innerHTML = `
            <div class="result-box">
                <p><strong>Status:</strong> ${data.status}</p>
                <p><strong>Rows Processed:</strong> ${data.rows_count ?? '—'}</p>
                <p><strong>Columns:</strong> ${data.cols_count ?? '—'}</p>
                <p><strong>Run ID:</strong> <code>${runId}</code></p>
                <p style="margin-top:12px;opacity:0.7;font-size:0.85rem;">
                    Full narration + clinical report stored in Supabase insights table.
                    Check the Agent Swarm Logs above for the complete synthesis.
                </p>
            </div>
        `;
    } catch (err) {
        logToConsole(`Failed to fetch final results: ${err.message}`, 'error');
    }
}

// ── Run pipeline ──────────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
    const mode = Array.from(modeRadios).find(r => r.checked)?.value || 'upload';

    if (mode === 'upload') {
        const hasCSV   = fileInput.files && fileInput.files.length > 0;
        const hasFasta = document.getElementById('fasta-input').files.length > 0;
        const hasPdf   = document.getElementById('pdf-input').files.length > 0;
        if (!hasCSV && !hasFasta && !hasPdf) {
            alert('Please select at least one file (CSV, FASTA, or PDF) before running.');
            return;
        }
    }

    runBtn.disabled    = true;
    runBtn.textContent = 'Initializing...';
    consoleOutput.innerHTML = '';
    resultsContent.innerHTML = '<div class="empty-state">⟳ Pipeline running — watch the logs...</div>';
    logToConsole('Starting Zora Multi-Agent Intelligence Pipeline...', 'system');

    const formData = new FormData();
    formData.append('mode', mode);

    if (mode === 'upload') {
        if (fileInput.files[0])                                formData.append('file',         fileInput.files[0]);
        const fasta = document.getElementById('fasta-input').files[0];
        const pdf   = document.getElementById('pdf-input').files[0];
        if (fasta)                                             formData.append('fasta_file',   fasta);
        if (pdf)                                               formData.append('pdf_file',     pdf);
        const target = document.getElementById('target-column').value.trim();
        if (target)                                            formData.append('target_column', target);
    }

    try {
        const response = await fetch(`${API_BASE}/api/run`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const result = await response.json();
        runIdDisplay.textContent = `Run: ${result.run_id}`;
        logToConsole(`✓ Run queued. Mode: ${result.mode || mode} | ID: ${result.run_id}`, 'success');
        connectSSE(result.run_id);

    } catch (err) {
        logToConsole(`✗ Failed to start run: ${err.message}`, 'error');
        runBtn.disabled    = false;
        runBtn.textContent = 'Initialize Pipeline';
    }
});
