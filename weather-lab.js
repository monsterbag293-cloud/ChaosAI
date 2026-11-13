/**
 * ChaosAI Project #1: Weather Pattern AI & NWP Model Comparator
 * Trains an in-browser MLP Neural Network on 365 days of Temperature + Cloud Coverage
 * and benchmarks its Next-Day Temperature prediction against a Numerical Weather Model.
 */

const WEATHER_LOCATIONS = {
  sf: {
    name: "San Francisco, CA",
    lat: 37.7749,
    lon: -122.4194,
    meanTemp: 14.8,
    ampTemp: 4.5,
    phaseShift: 225, // Peak warmth in late Aug/Sep
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
    phaseShift: 205, // Peak in late July/Aug
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
    phaseShift: 195, // Peak in July
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
  }
};

class WeatherMLP {
  constructor(inputSize = 7, hiddenSize = 12, lr = 0.015) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.lr = lr;

    // Xavier initialization
    const scale1 = Math.sqrt(2 / (inputSize + hiddenSize));
    const scale2 = Math.sqrt(2 / (hiddenSize + 1));

    this.W1 = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * scale1)
    );
    this.b1 = new Array(hiddenSize).fill(0);

    this.W2 = Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * scale2);
    this.b2 = 0;

    // Adam optimizer state
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
    const z1 = new Array(this.hiddenSize);
    const a1 = new Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.b1[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.W1[i][j] * x[j];
      }
      z1[i] = sum;
      a1[i] = Math.tanh(sum);
    }

    let out = this.b2;
    for (let i = 0; i < this.hiddenSize; i++) {
      out += this.W2[i] * a1[i];
    }
    return { out, a1, z1 };
  }

  trainBatch(samples) {
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

      // Output gradient
      const dout = (2 * err) / n;
      db2 += dout;
      for (let i = 0; i < this.hiddenSize; i++) {
        dW2[i] += dout * a1[i];
        const da1 = dout * this.W2[i];
        const dz1 = da1 * (1 - a1[i] * a1[i]); // tanh derivative
        db1[i] += dz1;
        for (let j = 0; j < this.inputSize; j++) {
          dW1[i][j] += dz1 * x[j];
        }
      }
    }

    // Adam update
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

class WeatherLabController {
  constructor() {
    this.locationKey = "sf";
    this.unit = "C"; // 'C' or 'F'
    this.dataset = []; // 365 days: { day, date, temp, cloud }
    this.nwpNextDayTemp = null; // Real weather model next-day prediction
    this.dataSourceLabel = "Physics Climate Generator (365 Days)";

    this.model = null;
    this.isTraining = false;
    this.lossHistory = [];
    this.testPredictions = []; // { day, actual, aiPred, nwpPred }
    this.aiNextDayPred = null;
    this.chaosNoise = 0; // 0..100%

    this.normStats = { tempMean: 15, tempStd: 5 };

    this.init();
  }

  init() {
    this.bindEvents();
    this.generateYearDataset(this.locationKey);
    this.drawYearChart();
    this.drawLossChart();
    this.drawCompareChart();
  }

  bindEvents() {
    const locSelect = document.getElementById("wLocationSelect");
    if (!locSelect) return;

    locSelect.addEventListener("change", (e) => {
      this.locationKey = e.target.value;
      this.resetModelState();
      this.generateYearDataset(this.locationKey);
      this.drawYearChart();
      this.drawCompareChart();
    });

    document.getElementById("wBtnGenerateData").addEventListener("click", () => {
      this.resetModelState();
      this.generateYearDataset(this.locationKey);
      this.drawYearChart();
      this.drawCompareChart();
      this.logWeather("Generated fresh 365-day temperature & cloud coverage series.", "info");
    });

    document.getElementById("wBtnFetchLive").addEventListener("click", () => {
      this.fetchLiveOpenMeteo(this.locationKey);
    });

    document.getElementById("wBtnTrain").addEventListener("click", () => {
      this.trainModel();
    });

    document.getElementById("wBtnTestNextDay").addEventListener("click", () => {
      this.evaluateNextDay();
    });

    document.getElementById("wUnitToggle").addEventListener("click", () => {
      this.unit = this.unit === "C" ? "F" : "C";
      document.getElementById("wUnitToggle").textContent = `Unit: °${this.unit}`;
      this.drawYearChart();
      this.drawCompareChart();
      this.updatePredictionCards();
    });

    const chaosSlider = document.getElementById("wChaosNoiseSlider");
    chaosSlider.addEventListener("input", (e) => {
      this.chaosNoise = parseInt(e.target.value, 10);
      document.getElementById("wChaosNoiseVal").textContent = `${this.chaosNoise}%`;
      if (this.model) {
        this.evaluateNextDay(false);
      }
    });

    window.addEventListener("resize", () => {
      this.drawYearChart();
      this.drawLossChart();
      this.drawCompareChart();
    });
  }

  formatTemp(tempC, showUnit = true) {
    const val = this.unit === "C" ? tempC : (tempC * 9) / 5 + 32;
    return `${val.toFixed(1)}${showUnit ? `°${this.unit}` : ""}`;
  }

  resetModelState() {
    this.model = null;
    this.isTraining = false;
    this.lossHistory = [];
    this.testPredictions = [];
    this.aiNextDayPred = null;
    document.getElementById("wBtnTestNextDay").disabled = true;
    document.getElementById("wAiPredVal").textContent = "--";
    document.getElementById("wNwpPredVal").textContent = "--";
    document.getElementById("wDeltaVal").textContent = "--";
    document.getElementById("wRmseVal").textContent = "--";
    document.getElementById("wTrainProgress").textContent = "Untrained";
    this.drawLossChart();
  }

  // Generates realistic 365-day temperature + cloud cover with physical coupling
  generateYearDataset(locKey) {
    const loc = WEATHER_LOCATIONS[locKey] || WEATHER_LOCATIONS.sf;
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

      // Seasonal baseline
      const seasonal = loc.meanTemp + loc.ampTemp * Math.cos(((d - loc.phaseShift) * 2 * Math.PI) / 365);
      // Synoptic 5-7 day weather front oscillation
      synopticWave = 0.72 * synopticWave + (Math.random() * 2 - 1) * loc.volatility;

      // Cloud cover with persistence + seasonal modulation
      const seasonalCloud = loc.baseCloud - loc.cloudAmp * Math.cos(((d - loc.phaseShift) * 2 * Math.PI) / 365);
      let cloud = 0.55 * prevCloud + 0.45 * seasonalCloud + (Math.random() * 2 - 1) * 24;
      cloud = Math.max(0, Math.min(100, cloud));

      // Temperature combines thermal inertia, seasonal forcing, synoptic front, and cloud radiative cooling
      const cloudEffect = (cloud - loc.baseCloud) * loc.cloudCoolingFactor;
      const targetTemp = 0.58 * prevTemp + 0.42 * (seasonal + synopticWave + cloudEffect) + (Math.random() * 2 - 1) * 0.65;

      this.dataset.push({
        day: d + 1,
        date: dateStr,
        temp: +targetTemp.toFixed(2),
        cloud: +cloud.toFixed(1)
      });

      prevTemp = targetTemp;
      prevCloud = cloud;
    }

    // Physical Numerical Weather Prediction (NWP) for Day 366
    const nextD = days;
    const seasonalNext = loc.meanTemp + loc.ampTemp * Math.cos(((nextD - loc.phaseShift) * 2 * Math.PI) / 365);
    const expectedCloudNext = Math.max(0, Math.min(100, 0.65 * prevCloud + 0.35 * loc.baseCloud));
    const cloudEffectNext = (expectedCloudNext - loc.baseCloud) * loc.cloudCoolingFactor;
    this.nwpNextDayTemp = +(0.58 * prevTemp + 0.42 * (seasonalNext + 0.7 * synopticWave + cloudEffectNext)).toFixed(2);

    this.dataSourceLabel = `Simulated Climate (${loc.name}, 365d)`;
    document.getElementById("wDataSourceBadge").textContent = this.dataSourceLabel;
    this.updateDatasetStats();
  }

  async fetchLiveOpenMeteo(locKey) {
    const loc = WEATHER_LOCATIONS[locKey] || WEATHER_LOCATIONS.sf;
    const btn = document.getElementById("wBtnFetchLive");
    btn.disabled = true;
    btn.textContent = "Fetching...";
    this.logWeather(`Querying Open-Meteo historical archive & NWP forecast for ${loc.name}...`, "info");

    try {
      const end = new Date();
      end.setDate(end.getDate() - 1);
      const start = new Date(end);
      start.setDate(end.getDate() - 364);

      const fmt = (d) => d.toISOString().split("T")[0];
      const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${loc.lat}&longitude=${loc.lon}&start_date=${fmt(start)}&end_date=${fmt(end)}&daily=temperature_2m_mean,cloud_cover_mean&timezone=auto`;
      const forecastUrl = `https://api.open-meteo.com/v1/forecast?latitude=${loc.lat}&longitude=${loc.lon}&daily=temperature_2m_mean&forecast_days=2&timezone=auto`;

      const [archRes, fcRes] = await Promise.all([fetch(archiveUrl), fetch(forecastUrl)]);
      if (!archRes.ok || !fcRes.ok) throw new Error("API response not OK");

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

      // Next-day real weather model prediction
      this.nwpNextDayTemp = fcData.daily.temperature_2m_mean[1] ?? fcData.daily.temperature_2m_mean[0];
      this.dataSourceLabel = `Live Open-Meteo (${loc.name}, ${this.dataset.length}d)`;
      document.getElementById("wDataSourceBadge").textContent = this.dataSourceLabel;

      this.resetModelState();
      this.updateDatasetStats();
      this.drawYearChart();
      this.drawCompareChart();
      this.logWeather(`Loaded ${this.dataset.length} real daily observations + live NWP next-day forecast (${this.formatTemp(this.nwpNextDayTemp)}).`, "success");
    } catch (err) {
      this.logWeather(`Live API unavailable in this environment (${err.message}). Using physics-coupled 365-day climate generator.`, "warn");
    } finally {
      btn.disabled = false;
      btn.textContent = "Fetch Live 365d (Open-Meteo)";
    }
  }

  updateDatasetStats() {
    if (!this.dataset.length) return;
    const temps = this.dataset.map(d => d.temp);
    const clouds = this.dataset.map(d => d.cloud);
    const meanT = temps.reduce((a, b) => a + b, 0) / temps.length;
    const stdT = Math.sqrt(temps.reduce((a, b) => a + (b - meanT) ** 2, 0) / temps.length) || 1;
    const meanC = clouds.reduce((a, b) => a + b, 0) / clouds.length;

    this.normStats = { tempMean: meanT, tempStd: stdT };
    document.getElementById("wStatDays").textContent = `${this.dataset.length} days`;
    document.getElementById("wStatMeanTemp").textContent = this.formatTemp(meanT);
    document.getElementById("wStatMeanCloud").textContent = `${meanC.toFixed(0)}%`;
  }

  buildFeatureVector(idx, injectNoise = false) {
    const d0 = this.dataset[idx];
    const d1 = this.dataset[Math.max(0, idx - 1)];
    const d2 = this.dataset[Math.max(0, idx - 2)];

    let t0 = d0.temp;
    let t1 = d1.temp;
    let t2 = d2.temp;
    let c0 = d0.cloud;
    let c1 = d1.cloud;

    if (injectNoise && this.chaosNoise > 0) {
      const noiseScale = this.chaosNoise / 100;
      t0 += (Math.random() * 2 - 1) * 6 * noiseScale;
      t1 += (Math.random() * 2 - 1) * 4 * noiseScale;
      c0 = Math.max(0, Math.min(100, c0 + (Math.random() * 2 - 1) * 50 * noiseScale));
    }

    const { tempMean, tempStd } = this.normStats;
    const angle = (2 * Math.PI * d0.day) / 365;

    return [
      (t0 - tempMean) / tempStd,
      (t1 - tempMean) / tempStd,
      (t2 - tempMean) / tempStd,
      (c0 - 50) / 30,
      (c1 - 50) / 30,
      Math.sin(angle),
      Math.cos(angle)
    ];
  }

  trainModel() {
    if (this.isTraining || this.dataset.length < 30) return;

    const epochs = parseInt(document.getElementById("wEpochsInput").value, 10) || 120;
    const hiddenSize = parseInt(document.getElementById("wHiddenInput").value, 10) || 12;
    const lr = parseFloat(document.getElementById("wLrInput").value) || 0.015;

    this.model = new WeatherMLP(7, hiddenSize, lr);
    this.lossHistory = [];
    this.isTraining = true;

    const trainBtn = document.getElementById("wBtnTrain");
    trainBtn.disabled = true;
    trainBtn.textContent = "Training AI...";

    // Split: Days 2..334 for training, Days 335..363 for 30-day holdout validation
    const splitIdx = this.dataset.length - 30;
    const trainSamples = [];
    for (let i = 2; i < splitIdx; i++) {
      trainSamples.push({
        x: this.buildFeatureVector(i, false),
        y: (this.dataset[i + 1].temp - this.normStats.tempMean) / this.normStats.tempStd
      });
    }

    this.logWeather(`Training WeatherMLP (7 -> ${hiddenSize} -> 1, Adam lr=${lr}) on ${trainSamples.length} days...`, "info");

    let currentEpoch = 0;
    const stepBatch = () => {
      const chunk = 4;
      for (let k = 0; k < chunk && currentEpoch < epochs; k++) {
        const mseNorm = this.model.trainBatch(trainSamples);
        const rmseC = Math.sqrt(mseNorm) * this.normStats.tempStd;
        this.lossHistory.push(rmseC);
        currentEpoch++;
      }

      const lastRmse = this.lossHistory[this.lossHistory.length - 1];
      document.getElementById("wTrainProgress").textContent = `Epoch ${currentEpoch}/${epochs} (RMSE: ${lastRmse.toFixed(2)}°C)`;
      this.drawLossChart();

      if (currentEpoch < epochs) {
        requestAnimationFrame(stepBatch);
      } else {
        this.isTraining = false;
        trainBtn.disabled = false;
        trainBtn.textContent = "Retrain AI Model";
        document.getElementById("wBtnTestNextDay").disabled = false;
        this.logWeather(`Training complete! Final training RMSE: ${lastRmse.toFixed(2)}°C.`, "success");
        this.evaluateNextDay(true);
      }
    };

    requestAnimationFrame(stepBatch);
  }

  evaluateNextDay(logResult = true) {
    if (!this.model) return;

    const { tempMean, tempStd } = this.normStats;
    const splitIdx = this.dataset.length - 30;
    this.testPredictions = [];

    let sqErrSum = 0;
    let count = 0;

    for (let i = splitIdx; i < this.dataset.length - 1; i++) {
      const x = this.buildFeatureVector(i, this.chaosNoise > 0);
      const predNorm = this.model.predict(x);
      const predTemp = predNorm * tempStd + tempMean;
      const actualTemp = this.dataset[i + 1].temp;

      sqErrSum += (predTemp - actualTemp) ** 2;
      count++;

      this.testPredictions.push({
        day: this.dataset[i + 1].day,
        date: this.dataset[i + 1].date,
        actual: actualTemp,
        aiPred: predTemp
      });
    }

    const holdoutRmse = Math.sqrt(sqErrSum / Math.max(1, count));

    // Predict Day 366 (Next Day)
    const lastIdx = this.dataset.length - 1;
    const nextX = this.buildFeatureVector(lastIdx, this.chaosNoise > 0);
    const nextPredNorm = this.model.predict(nextX);
    this.aiNextDayPred = +(nextPredNorm * tempStd + tempMean).toFixed(2);

    this.updatePredictionCards(holdoutRmse);
    this.drawCompareChart();

    if (logResult) {
      const delta = this.aiNextDayPred - this.nwpNextDayTemp;
      this.logWeather(
        `Next-Day Test -> AI: ${this.formatTemp(this.aiNextDayPred)} | Weather Model: ${this.formatTemp(this.nwpNextDayTemp)} | Delta: ${delta >= 0 ? "+" : ""}${delta.toFixed(2)}°C (30d RMSE: ${holdoutRmse.toFixed(2)}°C)`,
        Math.abs(delta) < 1.5 ? "success" : "warn"
      );

      // Record into ChaosAI Vault if app is available
      if (window.app && window.app.runs) {
        const resilience = Math.max(40, Math.min(99, Math.round(100 - Math.abs(delta) * 10 - holdoutRmse * 4)));
        window.app.runs.unshift({
          id: `WX-${Math.floor(1000 + Math.random() * 9000)}`,
          experimentId: "weather-ai-01",
          name: `Weather AI (${WEATHER_LOCATIONS[this.locationKey].name})`,
          category: "Weather Pattern AI",
          entropy: `${this.chaosNoise}%`,
          resilience: `${resilience}%`,
          recovery: `${holdoutRmse.toFixed(2)}°C RMSE`,
          tokens: this.dataset.length,
          status: resilience >= 75 ? "PASSED" : "DEGRADED",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
          date: new Date().toISOString().split("T")[0]
        });
        window.app.saveRunsToStorage();
        window.app.renderRunsTable();
      }
    }
  }

  updatePredictionCards(holdoutRmse) {
    if (this.aiNextDayPred === null) return;
    document.getElementById("wAiPredVal").textContent = this.formatTemp(this.aiNextDayPred);
    document.getElementById("wNwpPredVal").textContent = this.formatTemp(this.nwpNextDayTemp);

    const deltaC = this.aiNextDayPred - this.nwpNextDayTemp;
    const deltaDisplay = this.unit === "C" ? deltaC : (deltaC * 9) / 5;
    const deltaEl = document.getElementById("wDeltaVal");
    deltaEl.textContent = `${deltaDisplay >= 0 ? "+" : ""}${deltaDisplay.toFixed(2)}°${this.unit}`;
    deltaEl.style.color = Math.abs(deltaC) <= 1.5 ? "var(--chaos-emerald)" : "var(--chaos-amber)";

    if (holdoutRmse !== undefined) {
      const rmseDisplay = this.unit === "C" ? holdoutRmse : (holdoutRmse * 9) / 5;
      document.getElementById("wRmseVal").textContent = `±${rmseDisplay.toFixed(2)}°${this.unit}`;
    }
  }

  logWeather(msg, type = "info") {
    const box = document.getElementById("wLogOutput");
    if (!box) return;
    const div = document.createElement("div");
    div.className = `log-line text-${type}`;
    const ts = new Date().toISOString().substring(11, 19);
    div.textContent = `[${ts}] ${msg}`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  // --- CANVAS CHARTS ---
  drawYearChart() {
    const canvas = document.getElementById("wYearCanvas");
    if (!canvas || !this.dataset.length) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 44, r: 44, t: 20, b: 26 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    const temps = this.dataset.map(d => (this.unit === "C" ? d.temp : (d.temp * 9) / 5 + 32));
    const minT = Math.floor(Math.min(...temps) - 2);
    const maxT = Math.ceil(Math.max(...temps) + 2);

    // Grid lines
    ctx.strokeStyle = "#161e2e";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();

      const tLabel = (maxT - ((maxT - minT) * i) / 4).toFixed(0);
      ctx.fillStyle = "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.textAlign = "right";
      ctx.fillText(`${tLabel}°`, pad.l - 6, y + 3);
    }

    // Cloud coverage bars (0-100%)
    const barW = Math.max(1, plotW / this.dataset.length);
    ctx.fillStyle = "rgba(56, 189, 248, 0.20)";
    this.dataset.forEach((d, idx) => {
      const x = pad.l + (idx / (this.dataset.length - 1)) * plotW;
      const barH = (d.cloud / 100) * plotH * 0.65;
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

    // Train/Test split marker (Day 335)
    const splitX = pad.l + ((this.dataset.length - 30) / (this.dataset.length - 1)) * plotW;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#10b981";
    ctx.beginPath();
    ctx.moveTo(splitX, pad.t);
    ctx.lineTo(splitX, pad.t + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#10b981";
    ctx.textAlign = "left";
    ctx.fillText("30d Test Split", splitX + 4, pad.t + 12);
  }

  drawLossChart() {
    const canvas = document.getElementById("wLossCanvas");
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!this.lossHistory.length) {
      ctx.fillStyle = "#475569";
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Click 'Train AI Model' to visualize live epoch loss", w / 2, h / 2);
      return;
    }

    const pad = { l: 42, r: 16, t: 16, b: 24 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const maxL = Math.max(...this.lossHistory, 1);
    const minL = 0;

    ctx.strokeStyle = "#161e2e";
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (plotH * i) / 3;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.textAlign = "right";
      ctx.fillText((maxL - (maxL * i) / 3).toFixed(1), pad.l - 6, y + 3);
    }

    ctx.beginPath();
    this.lossHistory.forEach((loss, idx) => {
      const x = pad.l + (idx / Math.max(1, this.lossHistory.length - 1)) * plotW;
      const y = pad.t + plotH - ((loss - minL) / (maxL - minL)) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  drawCompareChart() {
    const canvas = document.getElementById("wCompareCanvas");
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) canvas.width = rect.width;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!this.testPredictions.length) {
      ctx.fillStyle = "#475569";
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Train AI to compare 30-day holdout & Day 366 forecast vs Weather Model", w / 2, h / 2);
      return;
    }

    const pad = { l: 42, r: 24, t: 18, b: 24 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    const toUnit = (c) => (this.unit === "C" ? c : (c * 9) / 5 + 32);
    const actuals = this.testPredictions.map(p => toUnit(p.actual));
    const preds = this.testPredictions.map(p => toUnit(p.aiPred));
    const nwpNext = toUnit(this.nwpNextDayTemp);
    const aiNext = toUnit(this.aiNextDayPred);

    const allVals = [...actuals, ...preds, nwpNext, aiNext];
    const minV = Math.floor(Math.min(...allVals) - 1.5);
    const maxV = Math.ceil(Math.max(...allVals) + 1.5);

    const totalPts = actuals.length + 1; // +1 for Day 366

    // Grid
    ctx.strokeStyle = "#161e2e";
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (plotH * i) / 3;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.font = "10px JetBrains Mono, monospace";
      ctx.textAlign = "right";
      ctx.fillText((maxV - ((maxV - minV) * i) / 3).toFixed(1) + "°", pad.l - 6, y + 3);
    }

    const getY = (v) => pad.t + plotH - ((v - minV) / Math.max(1, maxV - minV)) * plotH;
    const getX = (idx) => pad.l + (idx / Math.max(1, totalPts - 1)) * plotW;

    // Actuals line
    ctx.beginPath();
    actuals.forEach((v, i) => {
      if (i === 0) ctx.moveTo(getX(i), getY(v));
      else ctx.lineTo(getX(i), getY(v));
    });
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.stroke();

    // AI Predictions line
    ctx.beginPath();
    preds.forEach((v, i) => {
      if (i === 0) ctx.moveTo(getX(i), getY(v));
      else ctx.lineTo(getX(i), getY(v));
    });
    ctx.lineTo(getX(actuals.length), getY(aiNext));
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Day 366 markers
    const nextX = getX(actuals.length);
    // Weather model dot
    ctx.beginPath();
    ctx.arc(nextX, getY(nwpNext), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#f59e0b";
    ctx.fill();
    // AI dot
    ctx.beginPath();
    ctx.arc(nextX, getY(aiNext), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#06b6d4";
    ctx.fill();
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.weatherLab = new WeatherLabController();
});
