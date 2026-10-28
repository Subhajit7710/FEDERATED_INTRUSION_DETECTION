"use strict";

const state = {
  meta: null,
  sample: null,       // last sample from the server: {index, true_label, features}
  inputs: {},         // feature name -> <input>
  lastPredictionFor: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function fmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "";
  // drop float noise like 0.30000000000000004
  return String(Number(v.toPrecision(12)));
}

function pct(p) {
  return (p * 100).toFixed(p >= 0.9995 || p < 0.0005 ? 1 : 2) + "%";
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* ignore */ }
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`);
    err.body = body;
    throw err;
  }
  return body;
}

/* ---------- building the form ---------- */

function makeField(name, required) {
  const wrap = document.createElement("div");
  wrap.className = "field" + (required ? " required" : "");
  wrap.dataset.name = name.toLowerCase();

  const id = "f-" + state.meta.features.indexOf(name);
  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = name;
  label.title = name;

  const input = document.createElement("input");
  input.id = id;
  input.type = "text";
  input.inputMode = "decimal";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.dataset.feature = name;
  input.addEventListener("input", () => onFieldInput(name));
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") predict(); });

  wrap.append(label, input);
  state.inputs[name] = input;
  return wrap;
}

function buildForm() {
  const { features, empty_features } = state.meta;
  const req = $("required-grid");
  const pre = $("prefilled-grid");
  req.innerHTML = "";
  pre.innerHTML = "";
  for (const name of empty_features) req.append(makeField(name, true));
  for (const name of features) {
    if (!empty_features.includes(name)) pre.append(makeField(name, false));
  }
  $("pre-count").textContent = features.length - empty_features.length;
}

function buildSelects() {
  const m = $("model");
  m.innerHTML = "";
  for (const key of state.meta.models) {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = key.charAt(0).toUpperCase() + key.slice(1);
    if (key === state.meta.default_model) o.selected = true;
    m.append(o);
  }
  const k = $("kind");
  const group = document.createElement("optgroup");
  group.label = "Specific attack type";
  for (const lbl of state.meta.attack_labels) {
    const o = document.createElement("option");
    o.value = lbl;
    o.textContent = lbl;
    group.append(o);
  }
  k.append(group);
}

/* ---------- field state ---------- */

function isRequired(name) {
  return state.meta.empty_features.includes(name);
}

function parseValue(raw) {
  const s = raw.trim();
  if (s === "") return { empty: true };
  const v = Number(s);
  return Number.isFinite(v) ? { value: v } : { invalid: true };
}

function onFieldInput(name) {
  const input = state.inputs[name];
  const wrap = input.parentElement;
  const parsed = parseValue(input.value);
  wrap.classList.toggle("invalid", !!parsed.invalid);
  if (!isRequired(name) && state.sample) {
    const original = state.sample.features[name];
    const edited = parsed.empty || parsed.invalid || parsed.value !== Number(fmt(original));
    wrap.classList.toggle("edited", edited);
  }
  updateCounts();
}

function updateCounts() {
  const req = state.meta.empty_features;
  const filled = req.filter((n) => {
    const p = parseValue(state.inputs[n].value);
    return !p.empty && !p.invalid;
  }).length;
  $("req-count").textContent = `${filled} / ${req.length}`;
  $("edited-count").textContent = document.querySelectorAll(".field.edited").length;
}

function applySample(sample) {
  state.sample = sample;
  for (const [name, input] of Object.entries(state.inputs)) {
    const v = sample.features[name];
    const wrap = input.parentElement;
    wrap.classList.remove("edited", "invalid");
    if (isRequired(name)) {
      input.value = "";
      input.placeholder = "sample: " + fmt(v);
    } else {
      input.value = fmt(v);
    }
  }
  $("true-label").textContent = sample.true_label;
  $("sample-index").textContent = "#" + sample.index;
  updateCounts();
  clearResult();
}

/* ---------- actions ---------- */

async function refresh() {
  const btn = $("btn-refresh");
  btn.disabled = true;
  try {
    const kind = encodeURIComponent($("kind").value);
    const sample = await api(`/api/sample?kind=${kind}`);
    applySample(sample);
    setStatus("Loaded sample #" + sample.index);
  } catch (e) {
    setStatus("Could not load a sample: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function fillEmpty() {
  if (!state.sample) return;
  for (const name of state.meta.empty_features) {
    const input = state.inputs[name];
    if (input.value.trim() === "") {
      input.value = fmt(state.sample.features[name]);
      onFieldInput(name);
    }
  }
}

function undoEdits() {
  if (!state.sample) return;
  for (const [name, input] of Object.entries(state.inputs)) {
    if (isRequired(name)) continue;
    input.value = fmt(state.sample.features[name]);
    input.parentElement.classList.remove("edited", "invalid");
  }
  updateCounts();
}

function collectFeatures() {
  const features = {};
  const problems = [];
  for (const [name, input] of Object.entries(state.inputs)) {
    const p = parseValue(input.value);
    const wrap = input.parentElement;
    if (p.empty || p.invalid) {
      problems.push(name);
      wrap.classList.add("invalid");
    } else {
      features[name] = p.value;
    }
  }
  return { features, problems };
}

async function predict() {
  const { features, problems } = collectFeatures();
  if (problems.length) {
    showError(
      `${problems.length} field${problems.length > 1 ? "s need" : " needs"} a numeric value:`,
      problems
    );
    state.inputs[problems[0]].focus();
    return;
  }
  const btn = $("btn-predict");
  btn.disabled = true;
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, model: $("model").value }),
    });
    showResult(result);
    setStatus("Prediction done (" + result.model + " model)");
  } catch (e) {
    const d = e.body && e.body.detail;
    if (d && typeof d === "object") {
      showError(d.message || "Request rejected", [...(d.missing || []), ...(d.invalid || []), ...(d.features || [])]);
    } else {
      showError("Prediction failed: " + (d || e.message), []);
    }
    setStatus("Prediction failed", true);
  } finally {
    btn.disabled = false;
  }
}

/* ---------- result panel ---------- */

function clearResult() {
  $("result-body").className = "muted";
  $("result-body").textContent = "Fill the 6 empty fields and click Predict.";
}

function showError(message, names) {
  const body = $("result-body");
  body.className = "error-box";
  body.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = message;
  body.append(p);
  if (names.length) {
    const ul = document.createElement("ul");
    for (const n of names.slice(0, 12)) {
      const li = document.createElement("li");
      li.textContent = n;
      ul.append(li);
    }
    if (names.length > 12) {
      const li = document.createElement("li");
      li.textContent = `…and ${names.length - 12} more`;
      ul.append(li);
    }
    body.append(ul);
  }
}

function showResult(r) {
  const body = $("result-body");
  body.className = "";
  body.innerHTML = "";

  const isAttack = r.verdict === "ATTACK";
  const v = document.createElement("div");
  v.className = "verdict " + (isAttack ? "attack" : "benign");
  v.innerHTML = `<div class="label"></div>${isAttack ? '<div class="type"></div>' : ""}`;
  v.querySelector(".label").textContent = isAttack ? "ATTACK" : "BENIGN";
  if (isAttack) v.querySelector(".type").textContent = r.attack_type;
  body.append(v);

  const kv = document.createElement("dl");
  kv.className = "kv";
  const rows = [
    ["Attack probability (stage 1)", pct(r.attack_probability)],
    ["Decision threshold", pct(r.threshold)],
  ];
  if (isAttack) rows.push(["Type confidence (stage 2)", pct(r.attack_type_confidence)]);
  for (const [k, val] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = val;
    kv.append(dt, dd);
  }
  body.append(kv);

  if (isAttack && r.attack_type_probabilities) {
    const h = document.createElement("div");
    h.className = "muted";
    h.style.fontSize = "12px";
    h.style.marginBottom = "4px";
    h.textContent = "Attack type probabilities";
    body.append(h);

    const table = document.createElement("table");
    table.className = "probs";
    r.attack_type_probabilities.slice(0, 5).forEach((row, i) => {
      const tr = document.createElement("tr");
      if (i === 0) tr.className = "top";
      const name = document.createElement("td"); name.className = "name"; name.textContent = row.label; name.title = row.label;
      const barTd = document.createElement("td");
      barTd.innerHTML = '<div class="bar"><span></span></div>';
      barTd.querySelector("span").style.width = Math.max(0.5, row.probability * 100) + "%";
      const p = document.createElement("td"); p.className = "pct"; p.textContent = pct(row.probability);
      tr.append(name, barTd, p);
      table.append(tr);
    });
    body.append(table);
  }

  // compare with the sample's real label, only when nothing was changed
  if (state.sample) {
    const editedAny = document.querySelectorAll(".field.edited").length > 0;
    const requiredMatch = state.meta.empty_features.every(
      (n) => Number(state.inputs[n].value) === Number(fmt(state.sample.features[n]))
    );
    const m = document.createElement("div");
    if (!editedAny && requiredMatch) {
      const predicted = isAttack ? r.attack_type : "BENIGN";
      const ok = predicted === state.sample.true_label;
      m.className = "match " + (ok ? "ok" : "bad");
      m.textContent = ok
        ? `Matches the true label (${state.sample.true_label}).`
        : `True label is ${state.sample.true_label}.`;
    } else {
      m.className = "match muted";
      m.textContent = "Values differ from the original sample, so there is no true label to compare.";
    }
    body.append(m);
  }
}

/* ---------- filter ---------- */

function applyFilter() {
  const q = $("filter").value.trim().toLowerCase();
  for (const f of $("prefilled-grid").children) {
    f.classList.toggle("hidden", q !== "" && !f.dataset.name.includes(q));
  }
}

/* ---------- init ---------- */

async function init() {
  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    setStatus("Backend not reachable: " + e.message, true);
    return;
  }
  buildSelects();
  buildForm();
  $("btn-refresh").addEventListener("click", refresh);
  $("btn-fill").addEventListener("click", fillEmpty);
  $("btn-reset").addEventListener("click", undoEdits);
  $("btn-predict").addEventListener("click", predict);
  $("filter").addEventListener("input", applyFilter);
  $("model").addEventListener("change", () => { if ($("result-body").querySelector(".verdict")) predict(); });
  setStatus("Ready");
  await refresh();
}

init();
