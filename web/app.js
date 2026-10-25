"use strict";

const FIELDS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"];
const OPTIONAL = new Set(["SkinThickness", "Insulin", "DiabetesPedigreeFunction"]);
const LABELS = {
  Pregnancies: "Pregnancies", Glucose: "Glucose", BloodPressure: "Blood pressure", SkinThickness: "Skin thickness",
  Insulin: "Insulin", BMI: "BMI", DiabetesPedigreeFunction: "Diabetes pedigree function", Age: "Age",
};
// First row of the dataset; its insulin was not measured, so it is left blank and imputed.
const EXAMPLE = { Pregnancies: 6, Glucose: 148, BloodPressure: 72, SkinThickness: 35, Insulin: "", BMI: 33.6, DiabetesPedigreeFunction: 0.627, Age: 50 };
const WAKE_AFTER_MS = 2500;
const GIVE_UP_AFTER_MS = 120000;

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

// API base: ?api= wins, then window.API_BASE from config.js, else same origin.
function apiBase() {
  const fromQuery = new URLSearchParams(location.search).get("api");
  const raw = fromQuery || window.API_BASE || "";
  if (!raw) return "";
  try {
    const url = new URL(raw, location.href);
    if (url.protocol !== "https:" && url.protocol !== "http:") return "";
    return url.href.replace(/\/$/, "");
  } catch {
    return "";
  }
}
const API = apiBase();

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("is-error", isError);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// fetch that shows a "waking the server" note if the free host is cold, and retries while it boots.
async function fetchWaking(path, options = {}) {
  const started = Date.now();
  const wakeTimer = setTimeout(() => setStatus("Waking the server… free hosting sleeps when idle, so the first request can take up to a minute."), WAKE_AFTER_MS);
  try {
    for (;;) {
      try {
        const res = await fetch(API + path, options);
        if (![502, 503, 504].includes(res.status) || Date.now() - started > GIVE_UP_AFTER_MS) return res;
      } catch (err) {
        if (Date.now() - started > GIVE_UP_AFTER_MS) throw err;
      }
      await sleep(4000);
    }
  } finally {
    clearTimeout(wakeTimer);
  }
}

let threshold = null;

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

async function loadModelInfo() {
  try {
    const res = await fetchWaking("/model-info");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json();
    threshold = info.threshold;
    const t = info.metrics.test;
    $("model-line").innerHTML = "";
    const parts = [
      ["Held-out test ROC-AUC ", `${t.roc_auc.toFixed(3)} (95% CI ${t.roc_auc_ci95[0].toFixed(3)}–${t.roc_auc_ci95[1].toFixed(3)})`],
      [" · screening threshold ", info.threshold.toFixed(3)],
      [" · model ", info.model_name],
    ];
    for (const [label, value] of parts) {
      $("model-line").append(label);
      const b = document.createElement("strong");
      b.textContent = value;
      $("model-line").append(b);
    }
    setStatus(API ? `Using the model API at ${API}.` : "");
  } catch {
    $("model-line").textContent = "Model details could not be loaded.";
    setStatus("Could not reach the model API. Check your connection and reload the page.", true);
  }
}

function clearErrors() {
  for (const f of FIELDS) {
    $(f).removeAttribute("aria-invalid");
    $(`${f}-error`).textContent = "";
  }
}

function fieldError(name, message) {
  $(name).setAttribute("aria-invalid", "true");
  $(`${name}-error`).textContent = message;
}

function readForm() {
  const payload = {};
  let firstBad = null;
  for (const f of FIELDS) {
    const input = $(f);
    const raw = input.value.trim();
    if (raw === "") {
      if (OPTIONAL.has(f)) continue;
      fieldError(f, "Required.");
      firstBad = firstBad || input;
      continue;
    }
    const v = Number(raw);
    const min = Number(input.min), max = Number(input.max);
    if (!Number.isFinite(v) || v < min || v > max) {
      fieldError(f, `Enter a number from ${input.min} to ${input.max}.`);
      firstBad = firstBad || input;
      continue;
    }
    if (input.step === "1" && !Number.isInteger(v)) {
      fieldError(f, "Enter a whole number.");
      firstBad = firstBad || input;
      continue;
    }
    payload[f] = v;
  }
  return { payload, firstBad };
}

function showResult(r) {
  $("result").hidden = false;
  $("prob-value").textContent = pct(r.probability);
  $("meter-fill").style.width = pct(r.probability);
  $("meter-threshold").style.left = pct(r.threshold);
  const verdict = $("verdict");
  verdict.textContent = r.above_threshold
    ? `Above the screening threshold (${pct(r.threshold)}): worth discussing a proper test with a doctor.`
    : `Below the screening threshold (${pct(r.threshold)}).`;
  verdict.className = "verdict " + (r.above_threshold ? "is-above" : "is-below");
  $("imputed").textContent = r.imputed.length
    ? `Not provided, filled with the training median: ${r.imputed.map((f) => LABELS[f]).join(", ")}.`
    : "All eight measurements were provided.";
  $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
}

async function onSubmit(event) {
  event.preventDefault();
  clearErrors();
  const { payload, firstBad } = readForm();
  if (firstBad) {
    setStatus("Please fix the highlighted fields.", true);
    firstBad.focus();
    return;
  }
  const button = $("submit");
  button.disabled = true;
  button.textContent = "Estimating…";
  setStatus("Estimating…");
  try {
    const res = await fetchWaking("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.status === 422) {
      const body = await res.json().catch(() => ({}));
      for (const d of body.detail || []) {
        const name = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : null;
        if (FIELDS.includes(name)) fieldError(name, d.msg);
      }
      setStatus("The API rejected some values. Please check the highlighted fields.", true);
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showResult(await res.json());
    setStatus("");
  } catch {
    setStatus("Could not get an estimate: the model API did not respond. Please try again in a minute.", true);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate risk";
  }
}

function fillExample() {
  for (const f of FIELDS) $(f).value = EXAMPLE[f];
  clearErrors();
  setStatus("Example filled in (the first row of the dataset; insulin was not measured, so it is left blank).");
}

$("risk-form").addEventListener("submit", onSubmit);
$("example").addEventListener("click", fillExample);
if (API) {
  $("api-link").href = API + "/model-info";
  setStatus(`Using the model API at ${API}.`);
}
loadModelInfo();
