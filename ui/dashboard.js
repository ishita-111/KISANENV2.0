// dashboard.js
// KisanEnv 2.0 — Dashboard Controller (No launch overlay — inline Start/Stop)

import { updateFromFarmState } from './simulation.js';

let episodeRewards = [];
let currentEpisode = 0;
let ws = null;

// ═══ INIT ═════════════════════════════════════════════════════════════════

function init() {
  document.getElementById('start-btn').addEventListener('click', startSimulation);
  document.getElementById('stop-btn').addEventListener('click', stopSimulation);
}

// ═══ START / STOP ═════════════════════════════════════════════════════════

function startSimulation() {
  setStatus('RUNNING', 'badge-running');
  document.getElementById('start-btn').classList.add('hidden');
  document.getElementById('stop-btn').classList.remove('hidden');
  episodeRewards = [];
  currentEpisode = 0;
  connectWebSocket();
}

function stopSimulation() {
  if (ws) { ws.close(); ws = null; }
  setStatus('STOPPED', 'badge-complete');
  document.getElementById('stop-btn').classList.add('hidden');
  document.getElementById('start-btn').classList.remove('hidden');
}

function setStatus(text, cls) {
  const el = document.getElementById('sim-status');
  if (!el) return;
  el.textContent = text;
  el.className = 'sim-badge ' + cls;
}

// ═══ WEBSOCKET ════════════════════════════════════════════════════════════

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ max_episodes: 999999 }));
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'step') {
      updateFromFarmState(data.farm_state, data.weather || 'sunny');
      updateFarmPanel(data.farm_state, data.day);
      updateAgentPanel(data.action, data.reasoning, data.reward);
      updateOversightPanel(data.oversight);
      updateTimeline(data.day);
      flashReward(data.reward);
    }

    if (data.type === 'episode_complete') {
      episodeRewards.push(data.episode_reward);
      drawRewardChart();
      showReflection(data.reflection);
      currentEpisode++;
      setText('episode-counter', `Episode ${currentEpisode}`);
    }

    if (data.type === 'reset') {
      setText('episode-counter', `Episode ${currentEpisode + 1}`);
    }

    if (data.type === 'all_complete') {
      setStatus('COMPLETE', 'badge-complete');
      document.getElementById('stop-btn').classList.add('hidden');
      document.getElementById('start-btn').classList.remove('hidden');
    }
  };

  ws.onclose = () => {};
  ws.onerror = () => {};
}

// ═══ HUD UPDATES ══════════════════════════════════════════════════════════

function updateFarmPanel(fs, day) {
  setText('day-counter', `Day ${day} / 90`);
  setText('phase-label', getPhase(day));

  setBar('health-bar', fs.crop_health);
  setBar('moisture-bar', fs.soil_moisture);
  setBar('nitrogen-bar', fs.soil_nitrogen);
  setBar('soil-bar', fs.soil_health);
  setBar('pest-bar', fs.pest_pressure ?? fs.pest_pressure_observed ?? 0, true);
  setBar('fungal-bar', fs.fungal_risk, true);

  setText('budget-display', `Rs.${fs.budget.toLocaleString('en-IN')}`);

  const ins = document.getElementById('insurance-status');
  if (ins) {
    ins.textContent = fs.insurance_enrolled ? 'Insured' : 'Uninsured';
    ins.className = 'insurance-badge ' + (fs.insurance_enrolled ? 'badge-good' : 'badge-alert');
  }
}

function updateAgentPanel(action, reasoning, reward) {
  setText('action-display', action || '\u2014');

  const r = document.getElementById('reasoning-display');
  if (r) r.textContent = reasoning || '';

  const rw = document.getElementById('step-reward');
  if (rw) {
    rw.textContent = (reward >= 0 ? '+' : '') + reward.toFixed(3);
    rw.className = 'reward-text ' + (reward >= 0 ? 'reward-pos' : 'reward-neg');
  }
}

function updateOversightPanel(oversight) {
  if (!oversight) return;
  const score = oversight.score || 0;
  const el = document.getElementById('oversight-score');
  if (el) {
    el.textContent = `${(score * 100).toFixed(0)}%`;
    el.className = 'score-val ' + (score > 0.7 ? 'score-good' : score > 0.4 ? 'score-neutral' : 'score-poor');
  }
  setText('oversight-explanation', oversight.explanation || '');
  const sev = document.getElementById('oversight-severity');
  if (sev) {
    sev.textContent = (oversight.severity || 'neutral').toUpperCase();
    sev.className = 'severity-badge sev-' + (oversight.severity || 'neutral');
  }
}

function updateTimeline(day) {
  const el = document.getElementById('timeline-progress');
  if (el) el.style.width = `${(day / 90) * 100}%`;
}

function showReflection(text) {
  if (!text) return;
  const panel = document.getElementById('reflection-panel');
  const pre = document.getElementById('reflection-text');
  if (panel && pre) {
    pre.textContent = text;
    panel.classList.add('visible');
    setTimeout(() => panel.classList.remove('visible'), 12000);
  }
}

// ═══ REWARD CHART ═════════════════════════════════════════════════════════

function drawRewardChart() {
  const canvas = document.getElementById('reward-chart');
  if (!canvas || episodeRewards.length === 0) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.parentElement.clientWidth - 16;
  const H = canvas.height = 150;
  const pad = { top: 28, right: 12, bottom: 22, left: 38 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  ctx.fillStyle = '#101820';
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = '#d4a843';
  ctx.font = '12px Cormorant Garamond, serif';
  ctx.fillText('Training Progress', pad.left, 16);

  ctx.strokeStyle = 'rgba(200,160,60,0.1)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#5a6a7a';
    ctx.font = '8px IBM Plex Mono';
    ctx.fillText((1 - i * 0.25).toFixed(2), 2, y + 3);
  }

  const bY = pad.top + ph * (1 - 0.44);
  ctx.strokeStyle = '#e84040';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.left, bY); ctx.lineTo(W - pad.right, bY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#e84040';
  ctx.font = '7px IBM Plex Mono';
  ctx.fillText('Baseline 0.44', W - pad.right - 62, bY - 3);

  if (episodeRewards.length > 1) {
    const step = pw / Math.max(episodeRewards.length - 1, 1);

    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ph);
    episodeRewards.forEach((r, i) => {
      ctx.lineTo(pad.left + i * step, pad.top + ph * (1 - Math.min(r, 1)));
    });
    ctx.lineTo(pad.left + (episodeRewards.length - 1) * step, pad.top + ph);
    ctx.closePath();
    ctx.fillStyle = 'rgba(91,163,224,0.1)';
    ctx.fill();

    ctx.beginPath();
    ctx.strokeStyle = '#5ba3e0';
    ctx.lineWidth = 2;
    episodeRewards.forEach((r, i) => {
      const x = pad.left + i * step;
      const y = pad.top + ph * (1 - Math.min(r, 1));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    episodeRewards.forEach((r, i) => {
      ctx.beginPath();
      ctx.arc(pad.left + i * step, pad.top + ph * (1 - Math.min(r, 1)), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = '#d4a843';
      ctx.fill();
    });

    ctx.fillStyle = '#5a6a7a';
    ctx.font = '7px IBM Plex Mono';
    episodeRewards.forEach((_, i) => {
      if (i % Math.max(1, Math.floor(episodeRewards.length / 6)) === 0) {
        ctx.fillText(`Ep${i + 1}`, pad.left + i * step - 6, H - 4);
      }
    });
  }

  ctx.fillStyle = '#5ba3e0'; ctx.fillRect(W - 110, 6, 8, 3);
  ctx.fillStyle = '#b0c0d0'; ctx.font = '7px IBM Plex Mono'; ctx.fillText('Agent', W - 98, 10);
  ctx.fillStyle = '#e84040'; ctx.fillRect(W - 58, 6, 8, 3);
  ctx.fillText('Heuristic', W - 46, 10);
}

// ═══ UTILITIES ════════════════════════════════════════════════════════════

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(id, val, danger) {
  const el = document.getElementById(id);
  if (!el) return;
  const pct = Math.round(Math.min(val, 1) * 100);
  el.style.width = pct + '%';
  if (danger) {
    el.style.background = `hsl(${120 - pct}, 70%, 40%)`;
  } else {
    el.style.background = `hsl(${pct * 1.2}, 70%, 40%)`;
  }
}

function flashReward(reward) {
  // Disabled flickering
  return;
}

function getPhase(day) {
  if (day <= 15) return 'Setup & Insurance';
  if (day <= 45) return 'Growth Phase';
  if (day <= 70) return 'Critical Management';
  return 'Harvest & Sales';
}

init();

export { updateFarmPanel, updateAgentPanel, updateOversightPanel };
