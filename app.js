/**
 * ChaosAI Hub Application Controller & Visualization Engine
 */

class ChaosApp {
  constructor() {
    this.experiments = [...PRESET_EXPERIMENTS];
    this.currentExp = this.experiments[0];
    this.runs = this.loadRunsFromStorage();
    this.isSimulating = false;
    this.simulationTimer = null;
    this.simulationStepIdx = 0;
    
    // Canvas Visualization
    this.canvas = document.getElementById('chaosCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.animationFrameId = null;
    this.activeNodes = [];
    this.activeLinks = [];
    this.particles = [];

    this.init();
  }

  init() {
    this.bindNavigation();
    this.populateExperimentDropdown();
    this.bindControls();
    this.bindBuilder();
    this.bindVault();
    this.renderFaultVectors();
    this.loadExperiment(this.currentExp.id);
    this.initCanvas();
    this.renderRunsTable();
  }

  // --- NAVIGATION TABS ---
  bindNavigation() {
    const navButtons = document.querySelectorAll('.header-nav .nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        navButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const activePane = document.getElementById(
          targetTab === 'lab' ? 'tabLab' :
          targetTab === 'builder' ? 'tabBuilder' :
          targetTab === 'history' ? 'tabHistory' : 'tabMatrix'
        );
        if (activePane) activePane.classList.add('active');

        if (targetTab === 'lab') {
          this.resizeCanvas();
          this.renderCanvas();
        }
      });
    });

    document.getElementById('btnQuickSample').addEventListener('click', () => {
      const randomExp = this.experiments[Math.floor(Math.random() * this.experiments.length)];
      document.getElementById('experimentSelector').value = randomExp.id;
      this.loadExperiment(randomExp.id);
      document.getElementById('navLabBtn').click();
    });
  }

  // --- EXPERIMENT SELECTION & SYNC ---
  populateExperimentDropdown() {
    const select = document.getElementById('experimentSelector');
    select.innerHTML = '';
    this.experiments.forEach(exp => {
      const option = document.createElement('option');
      option.value = exp.id;
      option.textContent = `${exp.name} (${exp.category})`;
      select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
      this.loadExperiment(e.target.value);
    });
  }

  loadExperiment(expId) {
    const exp = this.experiments.find(x => x.id === expId) || this.experiments[0];
    this.currentExp = exp;

    document.getElementById('expCategoryBadge').textContent = exp.category;
    document.getElementById('expTitle').textContent = exp.name;
    document.getElementById('expDescription').textContent = exp.description;

    const tagsContainer = document.getElementById('expTags');
    tagsContainer.innerHTML = '';
    exp.tags.forEach(t => {
      const span = document.createElement('span');
      span.className = 'tag';
      span.textContent = t;
      tagsContainer.appendChild(span);
    });

    // Sync sliders
    const entropySlider = document.getElementById('entropySlider');
    entropySlider.value = exp.baseEntropy;
    document.getElementById('entropyLevelDisplay').textContent = `Entropy: ${exp.baseEntropy}%`;

    const latencySlider = document.getElementById('latencySlider');
    latencySlider.value = exp.latency;
    document.getElementById('latencyDisplay').textContent = `${exp.latency}ms ± 80ms`;

    const hallucinationSlider = document.getElementById('hallucinationPressure');
    hallucinationSlider.value = exp.hallucinationRate;
    document.getElementById('pressureDisplay').textContent = `${exp.hallucinationRate}% perturbation`;

    // Fault Toggles
    if (exp.faults) {
      document.getElementById('faultContextPrune').checked = !!exp.faults.contextPrune;
      document.getElementById('faultToolCorrupt').checked = !!exp.faults.toolCorrupt;
      document.getElementById('faultByzantine').checked = !!exp.faults.byzantine;
    }

    // Set Graph Architecture
    this.setupArchitecture(exp.architecture);
    this.resetLabState();
  }

  setupArchitecture(arch) {
    if (!arch || !arch.nodes) {
      this.activeNodes = [
        { id: "planner", label: "Planner / Core LLM", x: 200, y: 170, role: "agent", status: "healthy" },
        { id: "tool", label: "Tool Execution Gateway", x: 450, y: 170, role: "tool", status: "healthy" },
        { id: "arbiter", label: "Safety Arbiter", x: 650, y: 170, role: "arbiter", status: "healthy" }
      ];
      this.activeLinks = [
        { from: "planner", to: "tool" },
        { from: "tool", to: "arbiter" }
      ];
      return;
    }

    this.activeNodes = arch.nodes.map(n => ({
      ...n,
      status: "healthy",
      pulse: 0
    }));
    this.activeLinks = [...arch.links];
  }

  // --- CONTROLS & EXPERIMENT RUNNER ---
  bindControls() {
    const entropySlider = document.getElementById('entropySlider');
    entropySlider.addEventListener('input', (e) => {
      document.getElementById('entropyLevelDisplay').textContent = `Entropy: ${e.target.value}%`;
    });

    const latencySlider = document.getElementById('latencySlider');
    latencySlider.addEventListener('input', (e) => {
      document.getElementById('latencyDisplay').textContent = `${e.target.value}ms ± 80ms`;
    });

    const hallSlider = document.getElementById('hallucinationPressure');
    hallSlider.addEventListener('input', (e) => {
      document.getElementById('pressureDisplay').textContent = `${e.target.value}% perturbation`;
    });

    document.getElementById('btnRunExperiment').addEventListener('click', () => {
      this.startExperimentSimulation();
    });

    document.getElementById('btnResetLab').addEventListener('click', () => {
      this.resetLabState();
    });

    document.getElementById('btnClearLogs').addEventListener('click', () => {
      const term = document.getElementById('terminalOutput');
      term.innerHTML = '<div class="log-line text-muted">[System] Log buffer cleared. Ready.</div>';
    });

    document.getElementById('btnExportLog').addEventListener('click', () => {
      this.exportLogsToJson();
    });
  }

  resetLabState() {
    if (this.simulationTimer) clearInterval(this.simulationTimer);
    this.isSimulating = false;
    this.simulationStepIdx = 0;
    
    document.getElementById('btnRunExperiment').disabled = false;
    document.getElementById('btnRunExperiment').innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Run Experiment
    `;
    document.getElementById('canvasOverlay').style.display = 'block';

    // Reset KPIs
    document.getElementById('kpiResilience').textContent = '92%';
    document.getElementById('kpiHallucination').textContent = '0.14';
    document.getElementById('kpiTokens').textContent = '0';
    document.getElementById('kpiRecovery').textContent = '---';

    // Reset Node Statuses
    this.activeNodes.forEach(n => n.status = 'healthy');
    this.particles = [];
  }

  startExperimentSimulation() {
    if (this.isSimulating) return;
    this.isSimulating = true;
    this.simulationStepIdx = 0;
    document.getElementById('btnRunExperiment').disabled = true;
    document.getElementById('btnRunExperiment').textContent = 'Simulation Running...';
    document.getElementById('canvasOverlay').style.display = 'none';

    this.logToTerminal(`[START] Executing scenario: "${this.currentExp.name}" under ${document.getElementById('entropySlider').value}% Entropy`, 'text-info');

    const steps = this.currentExp.simulationSteps || [
      { step: 1, type: "info", text: "Starting autonomous agent loop.", targetNode: this.activeNodes[0]?.id, status: "healthy" },
      { step: 2, type: "warn", text: "[CHAOS] Perturbation injected into reasoning graph.", targetNode: this.activeNodes[1]?.id, status: "failing" },
      { step: 3, type: "success", text: "Heuristic fallback self-healed node state.", targetNode: this.activeNodes[1]?.id, status: "recovered" },
      { step: 4, type: "success", text: "Execution converged cleanly.", targetNode: this.activeNodes[0]?.id, status: "healthy" }
    ];

    let currentResilience = 98;
    let accumulatedTokens = 650;
    let currentHallucination = 0.08;

    this.simulationTimer = setInterval(() => {
      if (this.simulationStepIdx >= steps.length) {
        clearInterval(this.simulationTimer);
        this.completeExperimentRun(currentResilience, accumulatedTokens);
        return;
      }

      const curStep = steps[this.simulationStepIdx];
      const timestamp = new Date().toISOString().substring(11, 19);

      // Terminal entry
      this.logToTerminal(`[${timestamp}] [Step ${curStep.step}/${steps.length}] ${curStep.text}`, `text-${curStep.type}`);

      // Update Node Visual State
      if (curStep.targetNode) {
        const target = this.activeNodes.find(n => n.id === curStep.targetNode);
        if (target) {
          target.status = curStep.status;
          target.pulse = 1.0;
          this.emitParticles(target.x, target.y, curStep.status);
        }
      }

      // Telemetry Metric Drift
      if (curStep.status === 'failing' || curStep.type === 'error') {
        currentResilience = Math.max(50, currentResilience - Math.floor(Math.random() * 12 + 6));
        currentHallucination = +(currentHallucination + 0.12).toFixed(2);
      } else if (curStep.status === 'recovered') {
        currentResilience = Math.min(95, currentResilience + 6);
        currentHallucination = Math.max(0.05, +(currentHallucination - 0.08).toFixed(2));
      }
      accumulatedTokens += Math.floor(Math.random() * 450 + 200);

      document.getElementById('kpiResilience').textContent = `${currentResilience}%`;
      document.getElementById('kpiHallucination').textContent = `${currentHallucination}`;
      document.getElementById('kpiTokens').textContent = accumulatedTokens.toLocaleString();
      document.getElementById('kpiRecovery').textContent = `${Math.floor(Math.random() * 180 + 120)}ms`;

      this.simulationStepIdx++;
    }, 1200);
  }

  completeExperimentRun(resilience, tokens) {
    this.isSimulating = false;
    document.getElementById('btnRunExperiment').disabled = false;
    document.getElementById('btnRunExperiment').innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Run Experiment
    `;

    const status = resilience > 75 ? 'PASSED' : 'DEGRADED';
    const statusClass = resilience > 75 ? 'text-success' : 'text-warn';
    this.logToTerminal(`[COMPLETE] Simulation benchmark concluded. Final Resilience: ${resilience}%, Verdict: ${status}`, statusClass);

    // Save record to vault
    const runRecord = {
      id: `RUN-${Math.floor(1000 + Math.random() * 9000)}`,
      experimentId: this.currentExp.id,
      name: this.currentExp.name,
      category: this.currentExp.category,
      entropy: `${document.getElementById('entropySlider').value}%`,
      resilience: `${resilience}%`,
      recovery: document.getElementById('kpiRecovery').textContent,
      tokens: tokens,
      status: status,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      date: new Date().toISOString().split('T')[0]
    };

    this.runs.unshift(runRecord);
    this.saveRunsToStorage();
    this.renderRunsTable();
  }

  logToTerminal(text, cls = 'text-info') {
    const term = document.getElementById('terminalOutput');
    const line = document.createElement('div');
    line.className = `log-line ${cls}`;
    line.textContent = text;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
  }

  exportLogsToJson() {
    const logLines = Array.from(document.querySelectorAll('#terminalOutput .log-line')).map(l => l.textContent);
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
      experiment: this.currentExp.name,
      timestamp: new Date().toISOString(),
      logs: logLines
    }, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", `chaosai-log-${Date.now()}.json`);
    dlAnchor.click();
  }

  // --- CANVAS TOPOLOGY VISUALIZATION ---
  initCanvas() {
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    this.startAnimationLoop();
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) {
      this.canvas.width = rect.width;
      this.canvas.height = 340;
    }
  }

  emitParticles(x, y, status) {
    const color = status === 'failing' ? '#ef4444' :
                  status === 'stressed' ? '#f59e0b' :
                  status === 'recovered' ? '#06b6d4' : '#10b981';
    for (let i = 0; i < 14; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 2.5 + 1;
      this.particles.push({
        x: x,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1.0,
        color: color
      });
    }
  }

  startAnimationLoop() {
    const loop = () => {
      this.updatePhysics();
      this.renderCanvas();
      this.animationFrameId = requestAnimationFrame(loop);
    };
    loop();
  }

  updatePhysics() {
    // Update node pulses
    this.activeNodes.forEach(node => {
      if (node.pulse > 0) node.pulse = Math.max(0, node.pulse - 0.03);
    });

    // Update particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= 0.025;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  renderCanvas() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    // Subtle background grid
    ctx.strokeStyle = '#0e1420';
    ctx.lineWidth = 1;
    const gridSize = 32;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Render Links
    this.activeLinks.forEach(link => {
      const fromNode = this.activeNodes.find(n => n.id === link.from);
      const toNode = this.activeNodes.find(n => n.id === link.to);
      if (!fromNode || !toNode) return;

      ctx.beginPath();
      ctx.moveTo(fromNode.x, fromNode.y);
      ctx.lineTo(toNode.x, toNode.y);

      // Gradient stroke
      const grad = ctx.createLinearGradient(fromNode.x, fromNode.y, toNode.x, toNode.y);
      grad.addColorStop(0, '#1e293b');
      grad.addColorStop(0.5, '#334155');
      grad.addColorStop(1, '#1e293b');

      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Small animated energy pulse along link
      const time = Date.now() * 0.002;
      const progress = (time % 1);
      const px = fromNode.x + (toNode.x - fromNode.x) * progress;
      const py = fromNode.y + (toNode.y - fromNode.y) * progress;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
    });

    // Render Particles
    this.particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3 * p.life, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.life;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    });

    // Render Nodes
    this.activeNodes.forEach(node => {
      const radius = 22;
      const color = node.status === 'healthy' ? '#10b981' :
                    node.status === 'stressed' ? '#f59e0b' :
                    node.status === 'failing' ? '#ef4444' : '#06b6d4';

      // Pulse ring
      if (node.pulse > 0) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + (1 - node.pulse) * 18, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = node.pulse;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }

      // Outer glow circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = '#0f172a';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Inner dot
      ctx.beginPath();
      ctx.arc(node.x, node.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Label
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '600 11px Plus Jakarta Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(node.label, node.x, node.y + radius + 16);

      // Status sub-label
      ctx.fillStyle = color;
      ctx.font = '500 10px JetBrains Mono, monospace';
      ctx.fillText(node.status.toUpperCase(), node.x, node.y + radius + 29);
    });
  }

  // --- EXPERIMENT BUILDER ---
  bindBuilder() {
    const form = document.getElementById('builderForm');
    form.addEventListener('submit', (e) => {
      e.preventDefault();

      const name = document.getElementById('bExpName').value.trim();
      const category = document.getElementById('bExpCategory').value;
      const desc = document.getElementById('bExpDesc').value.trim();
      const entropy = parseInt(document.getElementById('bBaseEntropy').value, 10);
      const healStrategy = document.getElementById('bHealStrategy').value;
      const perturbationType = document.querySelector('input[name="perturbationType"]:checked')?.value || 'semantic';

      const newId = `exp-custom-${Date.now().toString().slice(-4)}`;
      const newExp = {
        id: newId,
        name: name,
        category: category,
        description: desc,
        tags: [category, `${perturbationType.toUpperCase()} Noise`, healStrategy],
        baseEntropy: entropy,
        latency: 350,
        hallucinationRate: entropy,
        faults: {
          contextPrune: true,
          toolCorrupt: perturbationType === 'protocol',
          byzantine: perturbationType === 'cognitive'
        },
        architecture: {
          nodes: [
            { id: "planner", label: "Autonomous Agent Core", x: 180, y: 170, role: "agent" },
            { id: "evaluator", label: "Critic & State Arbiter", x: 400, y: 110, role: "critic" },
            { id: "env_actor", label: "External Tool Interface", x: 620, y: 170, role: "tool" }
          ],
          links: [
            { from: "planner", to: "evaluator" },
            { from: "evaluator", to: "env_actor" },
            { from: "env_actor", to: "planner" }
          ]
        },
        simulationSteps: [
          { step: 1, type: "info", text: `Custom scenario launched: ${name}`, targetNode: "planner", status: "healthy" },
          { step: 2, type: "warn", text: `[CHAOS] Injected ${perturbationType} fault vector at ${entropy}% entropy level.`, targetNode: "env_actor", status: "stressed" },
          { step: 3, type: "error", text: "Anomaly detected in tool return payload; state drift observed.", targetNode: "env_actor", status: "failing" },
          { step: 4, type: "info", text: `Triggering configured mitigation: ${healStrategy}.`, targetNode: "evaluator", status: "stressed" },
          { step: 5, type: "success", text: "Self-healing converged: agent state reconciled with zero data loss.", targetNode: "planner", status: "recovered" },
          { step: 6, type: "success", text: "Benchmark concluded successfully.", targetNode: "planner", status: "healthy" }
        ]
      };

      this.experiments.unshift(newExp);
      this.populateExperimentDropdown();
      document.getElementById('experimentSelector').value = newId;
      this.loadExperiment(newId);

      // Switch to Lab tab
      document.getElementById('navLabBtn').click();
      this.logToTerminal(`[CUSTOM EXP CREATED] "${name}" is now ready for testing.`, 'text-success');

      form.reset();
    });
  }

  // --- EXPERIMENT VAULT / HISTORY ---
  loadRunsFromStorage() {
    const stored = localStorage.getItem('chaosai_vault_runs');
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (err) {
        console.error(err);
      }
    }
    // Default initial mock run entries for realism
    return [
      {
        id: "RUN-8921",
        experimentId: "exp-001",
        name: "Recursive Tool Denial & Loop Traps",
        category: "Autonomous Agent",
        entropy: "45%",
        resilience: "88%",
        recovery: "240ms",
        tokens: 4320,
        status: "PASSED",
        timestamp: "12:15:30",
        date: "2026-11-12"
      },
      {
        id: "RUN-8919",
        experimentId: "exp-002",
        name: "Adversarial Prompt Injection & Context Hijack",
        category: "Safety & Alignment",
        entropy: "70%",
        resilience: "94%",
        recovery: "180ms",
        tokens: 2840,
        status: "PASSED",
        timestamp: "11:42:10",
        date: "2026-11-12"
      },
      {
        id: "RUN-8894",
        experimentId: "exp-003",
        name: "Byzantine Multi-Agent Swarm Partition",
        category: "Multi-Agent System",
        entropy: "85%",
        resilience: "64%",
        recovery: "620ms",
        tokens: 8190,
        status: "DEGRADED",
        timestamp: "10:05:44",
        date: "2026-11-12"
      }
    ];
  }

  saveRunsToStorage() {
    localStorage.setItem('chaosai_vault_runs', JSON.stringify(this.runs));
    document.getElementById('historyCount').textContent = this.runs.length;
  }

  bindVault() {
    document.getElementById('historyCount').textContent = this.runs.length;

    document.getElementById('btnClearHistory').addEventListener('click', () => {
      if (confirm('Clear all historical benchmark runs from the vault?')) {
        this.runs = [];
        this.saveRunsToStorage();
        this.renderRunsTable();
      }
    });

    document.getElementById('btnExportAllCsv').addEventListener('click', () => {
      let csv = "Run ID,Experiment Name,Category,Entropy,Resilience,Recovery Latency,Tokens,Status,Date\n";
      this.runs.forEach(r => {
        csv += `"${r.id}","${r.name}","${r.category}","${r.entropy}","${r.resilience}","${r.recovery}",${r.tokens},"${r.status}","${r.date} ${r.timestamp}"\n`;
      });
      const dataUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
      const link = document.createElement('a');
      link.setAttribute('href', dataUri);
      link.setAttribute('download', `chaosai-benchmark-${Date.now()}.csv`);
      link.click();
    });

    // Modal Close
    document.getElementById('btnModalClose').addEventListener('click', () => {
      document.getElementById('runModal').classList.remove('active');
    });
    document.getElementById('btnModalDismiss').addEventListener('click', () => {
      document.getElementById('runModal').classList.remove('active');
    });
  }

  renderRunsTable() {
    const tbody = document.getElementById('runsTableBody');
    tbody.innerHTML = '';

    if (this.runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 24px; color: var(--text-muted);">No recorded runs in the vault yet. Execute an experiment in the Chaos Lab!</td></tr>`;
      return;
    }

    this.runs.forEach(r => {
      const tr = document.createElement('tr');
      const badgeCls = r.status === 'PASSED' ? 'badge-success' : 'badge-danger';
      tr.innerHTML = `
        <td class="font-mono text-info">${r.id}</td>
        <td><strong>${r.name}</strong></td>
        <td><span class="badge">${r.category}</span></td>
        <td class="font-mono">${r.entropy}</td>
        <td class="font-mono" style="font-weight:700; color: ${parseInt(r.resilience) > 75 ? 'var(--chaos-emerald)' : 'var(--chaos-amber)'}">${r.resilience}</td>
        <td class="font-mono">${r.recovery}</td>
        <td><span class="badge ${badgeCls}">${r.status}</span></td>
        <td class="font-mono text-muted" style="font-size:0.75rem;">${r.timestamp}</td>
        <td>
          <button class="btn btn-secondary btn-sm" style="padding: 4px 8px;" onclick="window.app.inspectRun('${r.id}')">Inspect</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  inspectRun(runId) {
    const run = this.runs.find(r => r.id === runId);
    if (!run) return;

    const modal = document.getElementById('runModal');
    document.getElementById('modalTitle').textContent = `Benchmark Telemetry: ${run.id}`;

    const modalContent = document.getElementById('modalContent');
    modalContent.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div>
          <h4 style="font-size: 1.05rem; margin-bottom: 4px;">${run.name}</h4>
          <span class="badge">${run.category}</span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; background: #0a0e16; padding:14px; border-radius: 8px;">
          <div><span class="sub-label">Entropy Injected:</span> <strong class="font-mono">${run.entropy}</strong></div>
          <div><span class="sub-label">Final Resilience:</span> <strong class="font-mono text-success">${run.resilience}</strong></div>
          <div><span class="sub-label">Recovery Latency:</span> <strong class="font-mono">${run.recovery}</strong></div>
          <div><span class="sub-label">Tokens Burned:</span> <strong class="font-mono">${run.tokens?.toLocaleString() || '4,200'}</strong></div>
          <div><span class="sub-label">Status Verdict:</span> <strong>${run.status}</strong></div>
          <div><span class="sub-label">Date & Time:</span> <span class="font-mono">${run.date} ${run.timestamp}</span></div>
        </div>
        <p style="font-size:0.84rem; color: var(--text-secondary); line-height:1.5;">
          The system evaluated autonomous loop recovery under configured adversarial noise. Safety arbiter confirmed no unbounded recursion occurred, and consensus quorum held within toleration thresholds.
        </p>
      </div>
    `;

    document.getElementById('btnModalReRun').onclick = () => {
      modal.classList.remove('active');
      const exp = this.experiments.find(e => e.id === run.experimentId) || this.experiments[0];
      document.getElementById('experimentSelector').value = exp.id;
      this.loadExperiment(exp.id);
      document.getElementById('navLabBtn').click();
    };

    modal.classList.add('active');
  }

  // --- FAULT VECTORS CATALOG ---
  renderFaultVectors() {
    const container = document.getElementById('vectorCatalog');
    container.innerHTML = '';

    FAULT_CATALOG.forEach(vec => {
      const card = document.createElement('div');
      card.className = 'vector-card';
      const sevBadge = vec.severity === 'Critical' ? 'badge-danger' :
                       vec.severity === 'High' ? 'badge-warning' : 'badge';

      card.innerHTML = `
        <div>
          <div class="vector-head">
            <span class="vector-code">${vec.code}</span>
            <span class="badge ${sevBadge}">${vec.severity}</span>
          </div>
          <h4 class="vector-title">${vec.title}</h4>
          <p class="vector-desc">${vec.description}</p>
        </div>
        <div class="vector-meta">
          <span>Category: <strong>${vec.category}</strong></span>
          <button class="btn btn-secondary btn-sm" style="padding: 3px 8px; font-size: 0.72rem;" onclick="window.app.applyFaultQuick('${vec.code}')">Test In Lab</button>
        </div>
      `;
      container.appendChild(card);
    });
  }

  applyFaultQuick(code) {
    // Quick stage a preset matching or set entropy
    const target = this.experiments[0];
    document.getElementById('experimentSelector').value = target.id;
    this.loadExperiment(target.id);
    document.getElementById('entropySlider').value = 75;
    document.getElementById('entropyLevelDisplay').textContent = 'Entropy: 75%';
    document.getElementById('navLabBtn').click();
    this.logToTerminal(`[FAULT APPLIED] Injected ${code} profile directly into current test harness.`, 'text-warn');
  }
}

// Global bootstrap
window.addEventListener('DOMContentLoaded', () => {
  window.app = new ChaosApp();
});
