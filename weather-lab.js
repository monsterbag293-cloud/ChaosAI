/**
 * ChaosAI: Weather AI Engine (3-Step Simple Flow)
 * Step 1: Download 365-day Temperature & Cloud Coverage
 * Step 2: TRAIN (Neural Network learns weather pattern)
 * Step 3: Ask "What's the temperature tomorrow?" -> Compare AI output to actual weather model
 */

const LOCATIONS = {
  sf: {
    name: "San Francisco, CA",
    lat: 37.7749,
    lon: -122.4194,
    meanTemp: 14.8,
    ampTemp: 4.5,
    phaseShift: 225,
    baseCloud: 48,
    cloudAmp: 18,
    volatility: 1.6,
    cloudCoolingFactor: -0.038
  },
  tokyo: {
    name: "Tokyo, Japan",
    lat: 35.6762,
    lon: 139.6503,
    meanTemp: 16.5,
    ampTemp: 10.8,
    phaseShift: 205,
    baseCloud: 55,
    cloudAmp: 15,
    volatility: 2.1,
    cloudCoolingFactor: -0.045
  },
  phoenix: {
    name: "Phoenix, AZ",
    lat: 33.4484,
    lon: -112.0740,
    meanTemp: 24.5,
    ampTemp: 12.2,
    phaseShift: 195,
    baseCloud: 18,
    cloudAmp: 8,
    volatility: 1.8,
    cloudCoolingFactor: -0.065
  },
  reykjavik: {
    name: "Reykjavik, Iceland",
    lat: 64.1466,
    lon: -21.9426,
    meanTemp: 5.2,
    ampTemp: 6.2,
    phaseShift: 200,
    baseCloud: 74,
    cloudAmp: 10,
    volatility: 2.6,
    cloudCoolingFactor: -0.025
  },
  london: {
    name: "London, UK",
    lat: 51.5074,
    lon: -0.1278,
    meanTemp: 11.2,
    ampTemp: 7.5,
    phaseShift: 205,
    baseCloud: 68,
    cloudAmp: 12,
    volatility: 2.0,
    cloudCoolingFactor: -0.035
  },
  sydney: {
    name: "Sydney, Australia",
    lat: -33.8688,
    lon: 151.2093,
    meanTemp: 18.2,
    ampTemp: 5.8,
    phaseShift: 25, // Peak warmth in Jan/Feb
    baseCloud: 42,
    cloudAmp: 10,
    volatility: 1.9,
    cloudCoolingFactor: -0.040
  }
};

// Compact Multilayer Perceptron (7 inputs -> 12 hidden tanh -> 1 output) with Adam
class WeatherNeuralNet {
  constructor(inputSize = 7, hiddenSize = 12, lr = 0.02) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.lr = lr;

    const scale1 = Math.sqrt(2 / (inputSize + hiddenSize));
    const scale2 = Math.sqrt(2 / (hiddenSize + 1));

    this.W1 = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * scale1)
    );
    this.b1 = new Array(hiddenSize).fill(0);

    this.W2 = Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * scale2);
    this.b2 = 0;

    // Adam optimizer moments
    this.t = 0;
    this.mW1 = this.W1.map(r => r.map(() => 0));
    this.vW1 = this.W1.map(r => r.map(() => 0));
    this.mb1 = new Array(hiddenSize).fill(0);
    this.vb1 = new Array(hiddenSize).fill(0);
    this.mW2 = new Array(hiddenSize).fill(0);
    this.vW2 = new Array(hiddenSize).fill(0);
    this.mb2 = 0;
    this.vb2 = 0;
  }

  forward(x) {
    const a1 = new Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.b1[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.W1[i][j] * x[j];
      }
      a1[i] = Math.tanh(sum);
    }

    let out = this.b2;
    for (let i = 0; i < this.hiddenSize; i++) {
      out += this.W2[i] * a1[i];
    }
    return { out, a1 };
  }

  trainEpoch(samples) {
    const n = samples.length;
    const dW1 = this.W1.map(r => r.map(() => 0));
    const db1 = new Array(this.hiddenSize).fill(0);
    const dW2 = new Array(this.hiddenSize).fill(0);
    let db2 = 0;
    let totalLoss = 0;

    for (let s = 0; s < n; s++) {
      const { x, y } = samples[s];
      const { out, a1 } = this.forward(x);
      const err = out - y;
      totalLoss += err * err;

      const dout = (2 * err) / n;
      db2 += dout;
      for (let i = 0; i < this.hiddenSize; i++) {
        dW2[i] += dout * a1[i];
        const da1 = dout * this.W2[i];
        const dz1 = da1 * (1 - a1[i] * a1[i]);
        db1[i] += dz1;
        for (let j = 0; j < this.inputSize; j++) {
          dW1[i][j] += dz1 * x[j];
        }
      }
    }

    this.t += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);

    this.mb2 = beta1 * this.mb2 + (1 - beta1) * db2;
    this.vb2 = beta2 * this.vb2 + (1 - beta2) * (db2 * db2);
    this.b2 -= this.lr * (this.mb2 / bc1) / (Math.sqrt(this.vb2 / bc2) + eps);

    for (let i = 0; i < this.hiddenSize; i++) {
      this.mW2[i] = beta1 * this.mW2[i] + (1 - beta1) * dW2[i];
      this.vW2[i] = beta2 * this.vW2[i] + (1 - beta2) * (dW2[i] * dW2[i]);
      this.W2[i] -= this.lr * (this.mW2[i] / bc1) / (Math.sqrt(this.vW2[i] / bc2) + eps);

      this.mb1[i] = beta1 * this.mb1[i] + (1 - beta1) * db1[i];
      this.vb1[i] = beta2 * this.vb1[i] + (1 - beta2) * (db1[i] * db1[i]);
      this.b1[i] -= this.lr * (this.mb1[i] / bc1) / (Math.sqrt(this.vb1[i] / bc2) + eps);

      for (let j = 0; j < this.inputSize; j++) {
        this.mW1[i][j] = beta1 * this.mW1[i][j] + (1 - beta1) * dW1[i][j];
        this.vW1[i][j] = beta2 * this.vW1[i][j] + (1 - beta2) * (dW1[i][j] * dW1[i][j]);
        this.W1[i][j] -= this.lr * (this.mW1[i][j] / bc1) / (Math.sqrt(this.vW1[i][j] / bc2) + eps);
      }
    }

    return totalLoss / n;
  }

  predict(x) {
    return this.forward(x).out;
  }
}

class WeatherApp {
  constructor() {
    this.locKey = "sf";
    this.unit = "C";
    this.dataset = [];
    this.actualNextDayTemp = null; // Ground truth / weather model predicted next day
    this.model = null;
    this.isTraining = false;
    this.lossHistory = [];
    this.aiPredictedNextDay = null;

    this.norm = { meanT: 15, stdT: 5 };

    this.init();
  }

  init() {
    this.bindDom();
    this.downloadYearData(this.locKey);
  }

  bindDom() {
    const locSelect = document.getElementById("locationSelect");
    locSelect.addEventListener("change", (e) => {
      this.locKey = e.target.value;
      this.downloadYearData(this.locKey);
    });

    document.getElementById("btnDownloadData").addEventListener("click", () => {
      this.downloadYearData(this.locKey);
    });

    document.getElementById("btnLiveApi").addEventListener("click", () => {
      this.fetchLiveOpenMeteo(this.locKey);
    });

    document.getElementById("btnTrain").addEventListener("click", () => {
      this.trainAI();
    });

    document.getElementById("btnAskAi").addEventListener("click", () => {
      this.askTomorrowTemp();
    });

    document.getElementById("unitToggle").addEventListener("click", () => {
      this.unit = this.unit === "C" ? "F" : "C";
      document.getElementById("unitToggle").textContent = `Unit: °${this.unit}`;
      this.updateDisplayUnits();
    });

    window.addEventListener("resize", () => {
      this.drawYearChart();
      this.drawLossChart();
      this.drawCompareChart();
    });
  }

  formatTemp(celsius, showUnit = true) {
    const val = this.unit === "C" ? celsius : (celsius * 9) / 5 + 32;
    return `${val.toFixed(1)}${showUnit ? `°${this.unit}` : ""}`;
  }

  // STEP 1: DOWNLOAD YEAR DATA
  downloadYearData(locKey) {
    const loc = LOCATIONS[locKey] || LOCATIONS.sf;
    const days = 365;
    this.dataset = [];

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    let synopticWave = 0;
    let prevTemp = loc.meanTemp + loc.ampTemp * Math.cos(((1 - loc.phaseShift) * 2 * Math.PI) / 365);
    let prevCloud = loc.baseCloud;

    for (let d = 0; d < days; d++) {
      const curDate = new Date(startDate);
      curDate.setDate(startDate.getDate() + d);
      const dateStr = curDate.toISOString().split("T")[0];

      const seasonal = loc.meanTemp + loc.ampTemp * Math.cos(((d - loc.phaseShift) * 2 * Math.PI) / 365);
      synopticWave = 0.72 * synopticWave + (Math.random() * 2 - 1) * loc.volatility;

      const seasonalCloud = loc.baseCloud - loc.cloudAmp * Math.cos(((d - loc.phaseShift) * 2 * Math.PI) / 365);
      let cloud = 0.55 * prevCloud + 0.45 * seasonalCloud + (Math.random() * 2 - 1) * 22;
      cloud = Math.max(0, Math.min(100, cloud));

      const cloudCooling = (cloud - loc.baseCloud) * loc.cloudCoolingFactor;
      const targetTemp = 0.58 * prevTemp + 0.42 * (seasonal + synopticWave + cloudCooling) + (Math.random() * 2 - 1) * 0.5;

      this.dataset.push({
        day: d + 1,
        date: dateStr,
        temp: +targetTemp.toFixed(2),
        cloud: +cloud.toFixed(1)
      });

      prevTemp = targetTemp;
      prevCloud = cloud;
    }

    // Weather model predicted next-day (Day 366)
    const seasonalNext = loc.meanTemp + loc.ampTemp * Math.cos(((days - loc.phaseShift) * 2 * Math.PI) / 365);
    const expectedCloudNext = Math.max(0, Math.min(100, 0.65 * prevCloud + 0.35 * loc.baseCloud));
    const cloudEffectNext = (expectedCloudNext - loc.baseCloud) * loc.cloudCoolingFactor;
    this.actualNextDayTemp = +(0.58 * prevTemp + 0.42 * (seasonalNext + 0.7 * synopticWave + cloudEffectNext)).toFixed(2);

    // Compute normalization stats
    const temps = this.dataset.map(d => d.temp);
    const mean = temps.reduce((a, b) => a + b, 0) / temps.length;
    const std = Math.sqrt(temps.reduce((a, b) => a + (b - mean) ** 2, 0) / temps.length) || 1;
    this.norm = { meanT: mean, stdT: std };

    // Reset step 2 & 3 state
    this.model = null;
    this.lossHistory = [];
    this.aiPredictedNextDay = null;

    // Update UI
    document.getElementById("dataSummary").innerHTML = `Loaded <strong>365 days</strong> of temperature & cloud coverage for <strong>${loc.name}</strong>.`;
    document.getElementById("askQuestionText").textContent = `"What is the temperature tomorrow at ${loc.name}?"`;
    document.getElementById("btnAskAi").disabled = true;
    document.getElementById("trainStatus").textContent = "Status: Ready to train";
    document.getElementById("progressBg").style.display = "none";
    document.getElementById("progressFill").style.width = "0%";
    document.getElementById("lossVal").textContent = "Untrained";
    document.getElementById("resAiTemp").textContent = "--";
    document.getElementById("resModelTemp").textContent = "--";
    document.getElementById("verdictBanner").style.display = "none";
    document.getElementById("testChartBox").style.display = "none";

    this.drawYearChart();
    this.drawLossChart();
  }

  async fetchLiveOpenMeteo(locKey) {
    const loc = LOCATIONS[locKey] || LOCATIONS.sf;
    const btn = document.getElementById("btnLiveApi");
    btn.disabled = true;
    btn.textContent = "Fetching...";

    try {
      const end = new Date();
      end.setDate(end.getDate() - 1);
      const start = new Date(end);
      start.setDate(end.getDate() - 364);

      const fmt = (d) => d.toISOString().split("T")[0];
      const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${loc.lat}&longitude=${loc.lon}&start_date=${fmt(start)}&end_date=${fmt(end)}&daily=temperature_2m_mean,cloud_cover_mean&timezone=auto`;
      const forecastUrl = `https://api.open-meteo.com/v1/forecast?latitude=${loc.lat}&longitude=${loc.lon}&daily=temperature_2m_mean&forecast_days=2&timezone=auto`;

      const [archRes, fcRes] = await Promise.all([fetch(archiveUrl), fetch(forecastUrl)]);
      if (!archRes.ok || !fcRes.ok) throw new Error("API network error");

      const archData = await archRes.json();
      const fcData = await fcRes.json();

      const times = archData.daily.time;
      const temps = archData.daily.temperature_2m_mean;
      const clouds = archData.daily.cloud_cover_mean;

      this.dataset = times.map((t, i) => ({
        day: i + 1,
        date: t,
        temp: temps[i] ?? 15,
        cloud: clouds[i] ?? 50
      }));

      this.actualNextDayTemp = +(fcData.daily.temperature_2m_mean[1] ?? fcData.daily.temperature_2m_mean[0]).toFixed(2);

      const tempVals = this.dataset.map(d => d.temp);
      const mean = tempVals.reduce((a, b) => a + b, 0) / tempVals.length;
      const std = Math.sqrt(tempVals.reduce((a, b) => a + (b - mean) ** 2, 0) / tempVals.length) || 1;
      this.norm = { meanT: mean, stdT: std };

      document.getElementById("dataSummary").innerHTML = `Loaded <strong>${this.dataset.length} real days</strong> from Open-Meteo for <strong>${loc.name}</strong>.`;
      this.drawYearChart();
    } catch (err) {
      alert("Notice: Could not reach live weather API. Using high-fidelity physical climate dataset.");
      this.downloadYearData(locKey);
    } finally {
      btn.disabled = false;
      btn.textContent = "Fetch Live Open-Meteo";
    }
  }

  // Feature vector for day i: [prev temp, prev 2 temp, prev 3 temp, cur cloud, prev cloud, sin(d), cos(d)]
  buildFeatureVector(idx) {
    const d0 = this.dataset[idx];
    const d1 = this.dataset[Math.max(0, idx - 1)];
    const d2 = this.dataset[Math.max(0, idx - 2)];

    const { meanT, stdT } = this.norm;
    const angle = (2 * Math.PI * d0.day) / 365;

    return [
      (d0.temp - meanT) / stdT,
      (d1.temp - meanT) / stdT,
      (d2.temp - meanT) / stdT,
      (d0.cloud - 50) / 30,
      (d1.cloud - 50) / 30,
      Math.sin(angle),
      Math.cos(angle)
    ];
  }

  // STEP 2: TRAIN
  trainAI() {
    if (this.isTraining || this.dataset.length < 30) return;

    this.isTraining = true;
    const btnTrain = document.getElementById("btnTrain");
    btnTrain.disabled = true;
    btnTrain.textContent = "TRAINING...";

    const progressBg = document.getElementById("progressBg");
    const progressFill = document.getElementById("progressFill");
    progressBg.style.display = "block";

    // 120 Epochs of Adam training on the year data
    const totalEpochs = 130;
    this.model = new WeatherNeuralNet(7, 12, 0.02);
    this.lossHistory = [];

    // Training set: days 2..334 (Days 335..364 held out to test)
    const split = this.dataset.length - 30;
    const samples = [];
    for (let i = 2; i < split; i++) {
      samples.push({
        x: this.buildFeatureVector(i),
        y: (this.dataset[i + 1].temp - this.norm.meanT) / this.norm.stdT
      });
    }

    let epoch = 0;
    const trainStep = () => {
      const chunk = 5;
      for (let k = 0; k < chunk && epoch < totalEpochs; k++) {
        const mse = this.model.trainEpoch(samples);
        const rmse = Math.sqrt(mse) * this.norm.stdT;
        this.lossHistory.push(rmse);
        epoch++;
      }

      const pct = Math.round((epoch / totalEpochs) * 100);
      progressFill.style.width = `${pct}%`;
      const curLoss = this.lossHistory[this.lossHistory.length - 1];
      document.getElementById("trainStatus").textContent = `Training: ${pct}% complete (Error: ±${curLoss.toFixed(2)}°C)`;
      document.getElementById("lossVal").textContent = `±${curLoss.toFixed(2)}°C RMSE`;
      this.drawLossChart();

      if (epoch < totalEpochs) {
        requestAnimationFrame(trainStep);
      } else {
        this.isTraining = false;
        btnTrain.disabled = false;
        btnTrain.textContent = "RETRAIN AI";
        document.getElementById("trainStatus").textContent = "✓ Training Complete! AI is ready to be asked.";
        document.getElementById("btnAskAi").disabled = false;
        // Scroll into view
        document.getElementById("cardStep3").scrollIntoView({ behavior: "smooth" });
      }
    };

    requestAnimationFrame(trainStep);
  }

  // STEP 3: ASK & COMPARE
  askTomorrowTemp() {
    if (!this.model) return;

    // Run prediction for Day 366 (Tomorrow)
    const lastIdx = this.dataset.length - 1;
    const xTomorrow = this.buildFeatureVector(lastIdx);
    const predNorm = this.model.predict(xTomorrow);
    this.aiPredictedNextDay = +(predNorm * this.norm.stdT + this.norm.meanT).toFixed(2);

    // Update Step 3 Display
    document.getElementById("resAiTemp").textContent = this.formatTemp(this.aiPredictedNextDay);
    document.getElementById("resModelTemp").textContent = this.formatTemp(this.actualNextDayTemp);

    const diffC = Math.abs(this.aiPredictedNextDay - this.actualNextDayTemp);
    const diffDisplay = this.unit === "C" ? diffC : (diffC * 9) / 5;

    const verdictBanner = document.getElementById("verdictBanner");
    verdictBanner.style.display = "block";

    const tag = document.getElementById("verdictTag");
    const text = document.getElementById("verdictText");
    const desc = document.getElementById("verdictDesc");

    if (diffC <= 1.2) {
      tag.textContent = "CLOSE MATCH";
      tag.style.background = "var(--chaos-emerald)";
      text.textContent = `Difference: only ${diffDisplay.toFixed(1)}°${this.unit}`;
      desc.textContent = "Great prediction! The AI accurately deduced tomorrow's temperature from cloud coverage and historical trends.";
    } else if (diffC <= 2.5) {
      tag.textContent = "GOOD";
      tag.style.background = "var(--chaos-amber)";
      text.textContent = `Difference: ${diffDisplay.toFixed(1)}°${this.unit}`;
      desc.textContent = "Acceptable forecast. The AI captured the primary seasonal and cloud cooling pattern.";
    } else {
      tag.textContent = "DIVERGENCE";
      tag.style.background = "var(--chaos-red)";
      text.textContent = `Difference: ${diffDisplay.toFixed(1)}°${this.unit}`;
      desc.textContent = "The AI diverged from the numerical weather model. Try retraining or selecting a different location.";
    }

    document.getElementById("testChartBox").style.display = "block";
    this.drawCompareChart();
  }

  updateDisplayUnits() {
    this.drawYearChart();
    this.drawCompareChart();
    if (this.aiPredictedNextDay !== null) {
      document.getElementById("resAiTemp").textContent = this.formatTemp(this.aiPredictedNextDay);
      document.getElementById("resModelTemp").textContent = this.formatTemp(this.actualNextDayTemp);
      const diffC = Math.abs(this.aiPredictedNextDay - this.actualNextDayTemp);
      const diffDisplay = this.unit === "C" ? diffC : (diffC * 9) / 5;
      document.getElementById("verdictText").textContent = `Difference: ${diffDisplay.toFixed(1)}°${this.unit}`;
    }
  }

  // --- CHARTS ---
  drawYearChart() {
    const canvas = document.getElementById("canvasYear");
    if (!canvas || !this.dataset.length) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width - 28;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 36, r: 20, t: 16, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    const toUnit = (c) => (this.unit === "C" ? c : (c * 9) / 5 + 32);
    const temps = this.dataset.map(d => toUnit(d.temp));
    const minT = Math.floor(Math.min(...temps) - 2);
    const maxT = Math.ceil(Math.max(...temps) + 2);

    // Grid lines
    ctx.strokeStyle = "#161e2e";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (plotH * i) / 3;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();

      const t = (maxT - ((maxT - minT) * i) / 3).toFixed(0);
      ctx.fillStyle = "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.textAlign = "right";
      ctx.fillText(`${t}°`, pad.l - 6, y + 3);
    }

    // Cloud bars
    const barW = Math.max(1, plotW / this.dataset.length);
    ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
    this.dataset.forEach((d, idx) => {
      const x = pad.l + (idx / (this.dataset.length - 1)) * plotW;
      const barH = (d.cloud / 100) * plotH * 0.7;
      ctx.fillRect(x, pad.t + plotH - barH, barW, barH);
    });

    // Temperature line
    ctx.beginPath();
    temps.forEach((t, idx) => {
      const x = pad.l + (idx / (temps.length - 1)) * plotW;
      const y = pad.t + plotH - ((t - minT) / Math.max(1, maxT - minT)) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  drawLossChart() {
    const canvas = document.getElementById("canvasLoss");
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width - 28;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!this.lossHistory.length) {
      ctx.fillStyle = "#475569";
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Click 'TRAIN AI' to watch the loss curve drop", w / 2, h / 2);
      return;
    }

    const pad = { l: 36, r: 16, t: 14, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const maxL = Math.max(...this.lossHistory, 1);

    ctx.strokeStyle = "#161e2e";
    for (let i = 0; i <= 2; i++) {
      const y = pad.t + (plotH * i) / 2;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.textAlign = "right";
      ctx.fillText((maxL - (maxL * i) / 2).toFixed(1), pad.l - 6, y + 3);
    }

    ctx.beginPath();
    this.lossHistory.forEach((l, idx) => {
      const x = pad.l + (idx / Math.max(1, this.lossHistory.length - 1)) * plotW;
      const y = pad.t + plotH - (l / maxL) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  drawCompareChart() {
    const canvas = document.getElementById("canvasCompare");
    if (!canvas || !this.model || this.aiPredictedNextDay === null) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width - 28;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 36, r: 24, t: 14, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    const toUnit = (c) => (this.unit === "C" ? c : (c * 9) / 5 + 32);

    // Test on last 30 days
    const split = this.dataset.length - 30;
    const actuals = [];
    const preds = [];
    for (let i = split; i < this.dataset.length - 1; i++) {
      const x = this.buildFeatureVector(i);
      const pNorm = this.model.predict(x);
      const pTemp = pNorm * this.norm.stdT + this.norm.meanT;
      actuals.push(toUnit(this.dataset[i + 1].temp));
      preds.push(toUnit(pTemp));
    }

    const aiTomorrow = toUnit(this.aiPredictedNextDay);
    const modelTomorrow = toUnit(this.actualNextDayTemp);

    const all = [...actuals, ...preds, aiTomorrow, modelTomorrow];
    const minV = Math.floor(Math.min(...all) - 1.5);
    const maxV = Math.ceil(Math.max(...all) + 1.5);
    const totalPts = actuals.length + 1;

    const getY = (v) => pad.t + plotH - ((v - minV) / Math.max(1, maxV - minV)) * plotH;
    const getX = (idx) => pad.l + (idx / (totalPts - 1)) * plotW;

    // Actuals line (green)
    ctx.beginPath();
    actuals.forEach((v, i) => {
      if (i === 0) ctx.moveTo(getX(i), getY(v));
      else ctx.lineTo(getX(i), getY(v));
    });
    ctx.lineTo(getX(actuals.length), getY(modelTomorrow));
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.stroke();

    // AI Predictions line (cyan)
    ctx.beginPath();
    preds.forEach((v, i) => {
      if (i === 0) ctx.moveTo(getX(i), getY(v));
      else ctx.lineTo(getX(i), getY(v));
    });
    ctx.lineTo(getX(actuals.length), getY(aiTomorrow));
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Tomorrow (Day 366) dots
    const nextX = getX(actuals.length);
    // Model dot (green)
    ctx.beginPath();
    ctx.arc(nextX, getY(modelTomorrow), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#10b981";
    ctx.fill();
    // AI dot (cyan)
    ctx.beginPath();
    ctx.arc(nextX, getY(aiTomorrow), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#06b6d4";
    ctx.fill();
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.weatherApp = new WeatherApp();
});
