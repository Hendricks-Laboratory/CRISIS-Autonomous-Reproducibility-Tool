/* =============================================================================
   web/static/script.js

   Front-end logic for the GPR Research Platform.
   Manages the 4-step analysis flow: upload → feature engineering →
   mode/feature selection → kernel configuration & training → plotting.

   Sections:
     STATE                  — module-level variables shared across steps
     STEP NAVIGATION        — nextStep / prevStep, step visibility helpers
     UPLOAD                 — CSV upload and column population
     FEATURE ENGINEERING    — expression builder and column management
     MODE STEP (3.x)        — std/mean/both mode selection, classify UI,
                              drag-drop, auto-detect, log vars, output col
     KERNEL BUILDER HTML    — buildKernelHTML() using <template> in tool.html
     KERNEL CONTROLS        — type/struct buttons, tuning section toggles
     KERNEL CONFIG READ     — getKernelConfig() serializes UI to payload
     RUN GPR                — bounds validation, fetch /run_gpr, metrics display
     PLOTTING               — dropdown population, color scheme, generatePlot()
============================================================================= */

/* ==============================
   STATE
============================== */
let ALL_COLUMNS = [];
let ANALYSIS_MODE = null;   // 'std' | 'mean' | 'both'
let CURRENT_STEP = 1;
let SELECTED_COLOR = "default";
let LAST_PLOT_TYPE = null;

// Std GP state
let STD_NUM_COLS = [];
let STD_CAT_COLS = [];
let STD_LOG_VARS = [];
let STD_MEASUREMENT_COL = null;
let STD_CONTROL_VARS = [];
let STD_CATEGORY_COMBOS = [];
let STD_GP_TARGET = "mean";
let STD_DONE = false;

// Mean GP state
let MEAN_NUM_COLS = [];
let MEAN_CAT_COLS = [];
let MEAN_LOG_VARS = [];
let MEAN_OUTPUT_COL = null;
let MEAN_CONTROL_VARS = [];
let MEAN_CATEGORY_COMBOS = [];
let MEAN_AUTO_NUM = [];
let MEAN_AUTO_CAT = [];
let MEAN_AUTO_DROPPED = [];
let MEAN_MODE = null;
let MEAN_DONE = false;

/* ==============================
   STEP SYSTEM
============================== */
function showStep(n) {
    document.querySelectorAll(".step").forEach(s => s.style.display = "none");
    const map = { 1: "uploadStep", 2: "featureEngineeringStep", 3: "modeStep", 4: "trainingStep" };
    const el = document.getElementById(map[n]);
    if (el) el.style.display = "block";
    CURRENT_STEP = n;
}

function prevStep() {
    if (CURRENT_STEP <= 1) return;
    // Clear downstream state when going back
    if (CURRENT_STEP === 4) {
        clearTrainingState();
    }
    if (CURRENT_STEP === 3) {
        clearModeState();
    }
    showStep(CURRENT_STEP - 1);
}

function clearTrainingState() {
    STD_DONE = false;
    MEAN_DONE = false;
    LAST_PLOT_TYPE = null;
    const plotSection = document.getElementById("plotSection");
    if (plotSection) plotSection.style.display = "none";
    const plotArea = document.getElementById("plotArea");
    if (plotArea) plotArea.innerHTML = "";
    ["stdMetrics","meanMetrics","stdGprError","meanGprError"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = "";
    });
}

function clearModeState() {
    clearTrainingState();
    STD_NUM_COLS = []; STD_CAT_COLS = []; STD_LOG_VARS = [];
    STD_MEASUREMENT_COL = null; STD_CONTROL_VARS = []; STD_CATEGORY_COMBOS = [];
    MEAN_NUM_COLS = []; MEAN_CAT_COLS = []; MEAN_LOG_VARS = [];
    MEAN_OUTPUT_COL = null; MEAN_CONTROL_VARS = []; MEAN_CATEGORY_COMBOS = [];
    MEAN_AUTO_NUM = []; MEAN_AUTO_CAT = []; MEAN_AUTO_DROPPED = [];
    MEAN_MODE = null;
}

/* ==============================
   UPLOAD
============================== */
async function uploadFile() {
    const file = document.getElementById("fileInput").files[0];
    if (!file) { alert("Please select a file."); return; }
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    ALL_COLUMNS = data.columns;
    showStep(2);
}

/* ==============================
   FEATURE ENGINEERING
============================== */
async function addFeature() {
    const input = document.getElementById("featureInput");
    const value = input.value.trim();
    if (!value) return;
    try {
        const res = await fetch("/apply_feature", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ expression: value })
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }
        ALL_COLUMNS = data.columns;
        const li = document.createElement("li");
        li.textContent = value;
        document.getElementById("featureList").appendChild(li);
        input.value = "";
    } catch (err) {
        alert("Feature engineering failed.");
    }
}

/* ==============================
   VALIDATION HELPERS
============================== */
function markFieldError(elementId, message) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.classList.add("field-error");

    const errId = "err_" + elementId;
    if (!document.getElementById(errId)) {
        const span = document.createElement("span");
        span.className = "error-msg";
        span.id = errId;
        span.textContent = "* " + message;
        el.insertAdjacentElement("afterend", span);
    }

    // Auto-clear on interaction
    const clear = () => { clearFieldError(elementId); };
    el.addEventListener("change", clear, { once: true });
    el.addEventListener("input", clear, { once: true });
    el.addEventListener("drop", clear, { once: true });
}

function clearFieldError(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.remove("field-error");
    const err = document.getElementById("err_" + elementId);
    if (err) err.remove();
}

function clearAllErrors() {
    document.querySelectorAll(".field-error").forEach(el => el.classList.remove("field-error"));
    document.querySelectorAll(".error-msg").forEach(el => el.remove());
    const summary = document.getElementById("stepErrorSummary");
    if (summary) { summary.style.display = "none"; summary.textContent = ""; }
}

function showErrorSummary(msg) {
    const summary = document.getElementById("stepErrorSummary");
    if (!summary) return;
    summary.textContent = msg;
    summary.style.display = "block";
}

/* ==============================
   MODE STEP
============================== */
function goToModeStep() {
    clearModeState();
    initModeStep();
    showStep(3);
}

function initModeStep() {
    renderDraggableList("stdAvailableCols", ALL_COLUMNS);
    document.getElementById("stdReplicateCols").innerHTML = "";
    document.getElementById("stdClassifySection").style.display = "none";
    document.getElementById("stdLogSection").style.display = "none";
    document.getElementById("stdTargetSection").style.display = "none";

    const measSel = document.getElementById("stdMeasurementCol");
    measSel.innerHTML = '<option value="">-- Select --</option>';
    ALL_COLUMNS.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col; opt.textContent = col;
        measSel.appendChild(opt);
    });

    const meanOut = document.getElementById("meanOutputCol");
    meanOut.innerHTML = '<option value="">-- Select Output Column --</option>';
    ALL_COLUMNS.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col; opt.textContent = col;
        meanOut.appendChild(opt);
    });

    renderDraggableList("meanAvailableCols", ALL_COLUMNS);
    document.getElementById("meanManualNum").innerHTML = "";
    document.getElementById("meanManualCat").innerHTML = "";
    document.getElementById("meanLogSection").style.display = "none";
    document.getElementById("meanTargetSection").style.display = "none";
    document.getElementById("meanAutoSection").style.display = "none";
    document.getElementById("meanManualSection").style.display = "none";
}

function selectAnalysisMode(mode) {
    ANALYSIS_MODE = mode;
    ["std","mean","both"].forEach(m => {
        document.getElementById("modeBtn_" + m).classList.toggle("active", m === mode);
    });
    document.getElementById("stdSection").style.display = (mode === "std" || mode === "both") ? "block" : "none";
    document.getElementById("meanSection").style.display = (mode === "mean" || mode === "both") ? "block" : "none";
    watchReplicateCols();
}

function watchReplicateCols() {
    const target = document.getElementById("stdReplicateCols");
    if (target._observer) target._observer.disconnect();
    const observer = new MutationObserver(() => {
        const cols = getListItems("stdReplicateCols");
        if (cols.length > 0) {
            populateStdClassify(cols);
            document.getElementById("stdClassifySection").style.display = "block";
        }
    });
    observer.observe(target, { childList: true });
    target._observer = observer;
}

// Called when measurement col dropdown changes
function onMeasurementColChange() {
    STD_MEASUREMENT_COL = document.getElementById("stdMeasurementCol").value;
    if (!STD_MEASUREMENT_COL) return;

    // Add measurement col to unclassified zone if not already there
    const unclassified = document.getElementById("stdUnclassified");
    const allClassified = [
        ...getListItems("stdNumCols"),
        ...getListItems("stdCatCols"),
        ...getListItems("stdUnclassified")
    ];
    if (!allClassified.includes(STD_MEASUREMENT_COL)) {
        const li = document.createElement("li");
        li.innerHTML = `${STD_MEASUREMENT_COL} <span style="color:#94a3b8;font-size:11px;">(measurement)</span>`;
        li.dataset.col = STD_MEASUREMENT_COL;
        li.id = "stdcol_meas_" + STD_MEASUREMENT_COL;
        li.draggable = true;
        unclassified.appendChild(li);
        document.getElementById("stdClassifySection").style.display = "block";
    }
}

function populateStdClassify(cols) {
    const unclassified = document.getElementById("stdUnclassified");
    const existing = [
        ...getListItems("stdNumCols"),
        ...getListItems("stdCatCols"),
        ...getListItems("stdUnclassified")
    ];
    cols.filter(c => !existing.includes(c)).forEach(col => {
        const li = document.createElement("li");
        li.textContent = col;
        li.id = "stdcol_" + col;
        li.draggable = true;
        unclassified.appendChild(li);
    });
}

function confirmStdClassify() {
    clearFieldError("stdNumCols");
    clearFieldError("stdUnclassified");
    clearFieldError("stdMeasurementCol");

    STD_MEASUREMENT_COL = document.getElementById("stdMeasurementCol").value;

    const numItems = [...document.querySelectorAll("#stdNumCols li")];
    const catItems = [...document.querySelectorAll("#stdCatCols li")];

    const tempNum = numItems.map(li => li.dataset.col || li.textContent.split("(")[0].trim()).filter(c => c !== STD_MEASUREMENT_COL);
    const tempCat = catItems.map(li => li.dataset.col || li.textContent.split("(")[0].trim()).filter(c => c !== STD_MEASUREMENT_COL);

    let valid = true;

    if (!STD_MEASUREMENT_COL) {
        markFieldError("stdMeasurementCol", "Please select a measurement column first.");
        valid = false;
    }

    if (tempNum.length === 0 && tempCat.length === 0) {
        markFieldError("stdUnclassified", "Drag at least one replicate column into Numerical or Categorical.");
        valid = false;
    }

    if (!valid) return;

    STD_NUM_COLS = tempNum;
    STD_CAT_COLS = tempCat;

    renderLogVarCheckboxes("stdLogContainer", STD_NUM_COLS, "std");
    document.getElementById("stdLogSection").style.display = STD_NUM_COLS.length > 0 ? "block" : "none";
    document.getElementById("stdTargetSection").style.display = "block";
}

/* ==============================
   MEAN MODE FEATURE SELECTION
============================== */
function selectMeanMode(mode) {
    MEAN_MODE = mode;
    document.getElementById("meanAutoSection").style.display = "none";
    document.getElementById("meanManualSection").style.display = "none";
    if (mode === "auto") {
        initMeanAuto();
        document.getElementById("meanAutoSection").style.display = "block";
    } else {
        document.getElementById("meanManualSection").style.display = "block";
    }
}

async function initMeanAuto() {
    // Send empty target_col — backend returns all cols, we drop output col after selection
    const res = await fetch("/auto_detect_features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_col: "" })
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    MEAN_AUTO_NUM = [...data.num_cols];
    MEAN_AUTO_CAT = [...data.cat_cols];
    MEAN_AUTO_DROPPED = [];
    renderMeanAutoLists();
}

function onMeanOutputColChange() {
    MEAN_OUTPUT_COL = document.getElementById("meanOutputCol").value;
    if (!MEAN_OUTPUT_COL || MEAN_MODE !== "auto") return;

    // Drop chosen output col from feature lists
    if (MEAN_AUTO_NUM.includes(MEAN_OUTPUT_COL)) {
        MEAN_AUTO_NUM = MEAN_AUTO_NUM.filter(c => c !== MEAN_OUTPUT_COL);
        renderMeanAutoLists();
    } else if (MEAN_AUTO_CAT.includes(MEAN_OUTPUT_COL)) {
        MEAN_AUTO_CAT = MEAN_AUTO_CAT.filter(c => c !== MEAN_OUTPUT_COL);
        renderMeanAutoLists();
    }
}

function renderMeanAutoLists() {
    document.getElementById("meanAutoNum").innerHTML = "";
    document.getElementById("meanAutoCat").innerHTML = "";
    document.getElementById("meanAutoDropped").innerHTML = "";

    MEAN_AUTO_NUM.forEach(col => {
        document.getElementById("meanAutoNum").innerHTML +=
            `<li>${col} <button class="feature-btn" onclick="dropMeanFeature('${col}','num')">❌</button></li>`;
    });
    MEAN_AUTO_CAT.forEach(col => {
        document.getElementById("meanAutoCat").innerHTML +=
            `<li>${col} <button class="feature-btn" onclick="dropMeanFeature('${col}','cat')">❌</button></li>`;
    });
    MEAN_AUTO_DROPPED.forEach(obj => {
        document.getElementById("meanAutoDropped").innerHTML +=
            `<li>${obj.name} <button class="feature-btn restore-btn" onclick="restoreMeanFeature('${obj.name}','${obj.type}')">➕</button></li>`;
    });

    renderLogVarCheckboxes("meanLogContainer", MEAN_AUTO_NUM, "mean");
    document.getElementById("meanLogSection").style.display = MEAN_AUTO_NUM.length > 0 ? "block" : "none";
    document.getElementById("meanTargetSection").style.display = "block";
}

function dropMeanFeature(col, type) {
    if (type === "num") MEAN_AUTO_NUM = MEAN_AUTO_NUM.filter(c => c !== col);
    else MEAN_AUTO_CAT = MEAN_AUTO_CAT.filter(c => c !== col);
    MEAN_AUTO_DROPPED.push({ name: col, type });
    renderMeanAutoLists();
}

function restoreMeanFeature(col, type) {
    MEAN_AUTO_DROPPED = MEAN_AUTO_DROPPED.filter(o => o.name !== col);
    // Don't restore if it's currently the output col
    if (col === MEAN_OUTPUT_COL) return;
    if (type === "num") MEAN_AUTO_NUM.push(col);
    else MEAN_AUTO_CAT.push(col);
    renderMeanAutoLists();
}

function confirmMeanManual() {
    MEAN_NUM_COLS = getListItems("meanManualNum");
    MEAN_CAT_COLS = getListItems("meanManualCat");
    renderLogVarCheckboxes("meanLogContainer", MEAN_NUM_COLS, "mean");
    document.getElementById("meanLogSection").style.display = MEAN_NUM_COLS.length > 0 ? "block" : "none";
    document.getElementById("meanTargetSection").style.display = "block";
}

/* ==============================
   LOG VARS
============================== */
function renderLogVarCheckboxes(containerId, numCols, prefix) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    numCols.forEach(col => {
        const label = document.createElement("label");
        label.innerHTML = `<input type="checkbox" value="${col}" onchange="updateLogVars('${prefix}')"> ${col}`;
        container.appendChild(label);
        container.appendChild(document.createElement("br"));
    });
}

function updateLogVars(prefix) {
    const vars = [...document.querySelectorAll(`#${prefix}LogContainer input:checked`)].map(x => x.value);
    if (prefix === "std") STD_LOG_VARS = vars;
    else MEAN_LOG_VARS = vars;
}

/* ==============================
   GO TO TRAINING
============================== */
function goToTrainingStep() {
    clearAllErrors();
    let valid = true;

    // Mode must be selected
    if (!ANALYSIS_MODE) {
        markFieldError("modeSelector", "Please select an analysis mode.");
        showErrorSummary("Please fill in the required fields highlighted below.");
        return;
    }

    if (ANALYSIS_MODE === "std" || ANALYSIS_MODE === "both") {
        const replicateCols = getListItems("stdReplicateCols");
        const measCol = document.getElementById("stdMeasurementCol").value;
        const numCols = getListItems("stdNumCols");
        const catCols = getListItems("stdCatCols");

        if (replicateCols.length === 0) {
            markFieldError("stdReplicateCols", "Drag at least one replicate column here.");
            valid = false;
        }
        if (!measCol) {
            markFieldError("stdMeasurementCol", "Measurement column required.");
            valid = false;
        }
        if (replicateCols.length > 0 && numCols.length === 0 && catCols.length === 0) {
            markFieldError("stdClassifySection", "Please click Confirm Classification and classify your columns.");
            valid = false;
        }
    }

    if (ANALYSIS_MODE === "mean" || ANALYSIS_MODE === "both") {
        if (!MEAN_MODE) {
            markFieldError("meanFeatureModeButtons", "Please select Automatic or Manual.");
            valid = false;
        } else {
            const numCols = MEAN_MODE === "auto" ? MEAN_AUTO_NUM : getListItems("meanManualNum");
            const catCols = MEAN_MODE === "auto" ? MEAN_AUTO_CAT : getListItems("meanManualCat");
            if (numCols.length === 0 && catCols.length === 0) {
                const targetId = MEAN_MODE === "auto" ? "meanAutoSection" : "meanManualSection";
                markFieldError(targetId, "At least one feature required.");
                valid = false;
            }
        }
        const outputCol = document.getElementById("meanOutputCol").value;
        if (!outputCol) {
            markFieldError("meanOutputCol", "GP Target (output column) required.");
            valid = false;
        }
    }

    if (!valid) {
        showErrorSummary("Please fill in the required fields highlighted in red below.");
        return;
    }

    // Collect state
    if (ANALYSIS_MODE === "mean" || ANALYSIS_MODE === "both") {
        if (MEAN_MODE === "auto") {
            MEAN_NUM_COLS = [...MEAN_AUTO_NUM];
            MEAN_CAT_COLS = [...MEAN_AUTO_CAT];
        } else {
            MEAN_NUM_COLS = getListItems("meanManualNum");
            MEAN_CAT_COLS = getListItems("meanManualCat");
        }
        MEAN_OUTPUT_COL = document.getElementById("meanOutputCol").value;
        MEAN_LOG_VARS = [...document.querySelectorAll("#meanLogContainer input:checked")].map(x => x.value);
    }

    if (ANALYSIS_MODE === "std" || ANALYSIS_MODE === "both") {
        STD_MEASUREMENT_COL = document.getElementById("stdMeasurementCol").value;
        STD_NUM_COLS = [...document.querySelectorAll("#stdNumCols li")]
            .map(li => li.dataset.col || li.textContent.split("(")[0].trim())
            .filter(c => c !== STD_MEASUREMENT_COL);
        STD_CAT_COLS = [...document.querySelectorAll("#stdCatCols li")]
            .map(li => li.dataset.col || li.textContent.split("(")[0].trim())
            .filter(c => c !== STD_MEASUREMENT_COL);
        STD_LOG_VARS = [...document.querySelectorAll("#stdLogContainer input:checked")].map(x => x.value);
        const checked = document.querySelector('input[name="stdGpTarget"]:checked');
        STD_GP_TARGET = checked ? checked.value : "mean";
    }

    clearTrainingState();
    initTrainingStep();
    showStep(4);
}

function initTrainingStep() {
    const showStd  = ANALYSIS_MODE === "std"  || ANALYSIS_MODE === "both";
    const showMean = ANALYSIS_MODE === "mean" || ANALYSIS_MODE === "both";
    const isBoth   = ANALYSIS_MODE === "both";

    document.getElementById("stdGpBlock").style.display  = showStd  ? "block" : "none";
    document.getElementById("meanGpBlock").style.display = showMean ? "block" : "none";

    const trainingInner = document.getElementById("trainingInner");
    if (trainingInner) {
        trainingInner.className = isBoth ? "gp-blocks-both" : "gp-blocks-single";
    }

    if (showStd)  document.getElementById("stdKernelBuilder").innerHTML  = buildKernelHTML("std");
    if (showMean) document.getElementById("meanKernelBuilder").innerHTML = buildKernelHTML("mean");
}

/* ==============================
   KERNEL BUILDER HTML
============================== */
function buildKernelHTML(prefix) {
    const tpl = document.getElementById("kernelBuilderTemplate");
    return tpl.innerHTML.replaceAll("__P__", prefix);
}

function selectKernelType(prefix, type) {
    ["rbf","matern","rq"].forEach(t => {
        document.getElementById(`${prefix}_kernelBtn_${t}`).classList.toggle("active", t === type);
    });
    const nu = document.getElementById(`${prefix}_nuSection`);
    const rq = document.getElementById(`${prefix}_rqSection`);
    if (nu) nu.style.display = type === "matern" ? "block" : "none";
    if (rq) rq.style.display = type === "rq" ? "block" : "none";
}

function selectKernelStruct(prefix, struct) {
    ["constant","white"].forEach(s => {
        document.getElementById(`${prefix}_structBtn_${s}`).classList.toggle("active", s === struct);
    });
    const c = document.getElementById(`${prefix}_constantTuning`);
    const w = document.getElementById(`${prefix}_whiteTuning`);
    if (c) c.style.display = struct === "constant" ? "block" : "none";
    if (w) w.style.display = struct === "white" ? "block" : "none";
}

function toggleBaseTuning(prefix) {
    const checked = document.getElementById(`${prefix}_baseTuning`).checked;
    document.getElementById(`${prefix}_baseTuningSection`).style.display = checked ? "block" : "none";
}

function toggleStructTuning(prefix) {
    const checked = document.getElementById(`${prefix}_structTuning`).checked;
    document.getElementById(`${prefix}_structTuningSection`).style.display = checked ? "block" : "none";
}

function getKernelConfig(prefix) {
    const baseTuning   = document.getElementById(`${prefix}_baseTuning`).checked;
    const structTuning = document.getElementById(`${prefix}_structTuning`).checked;

    const kernelType = ["rbf","matern","rq"].find(t =>
        document.getElementById(`${prefix}_kernelBtn_${t}`)?.classList.contains("active")
    ) || "rbf";

    const struct = ["constant","white"].find(s =>
        document.getElementById(`${prefix}_structBtn_${s}`)?.classList.contains("active")
    ) || "constant";

    const config = { kernel_type: kernelType, white_noise: struct === "white", ard: true };

    if (baseTuning) {
        config.length_scale_init = parseFloat(document.getElementById(`${prefix}_lsInit`).value);
        config.length_scale_bounds = [
            parseFloat(document.getElementById(`${prefix}_lsLower`).value),
            parseFloat(document.getElementById(`${prefix}_lsUpper`).value)
        ];
        if (kernelType === "matern") {
            config.nu = parseFloat(document.getElementById(`${prefix}_nu`).value);
        }
        if (kernelType === "rq") {
            config.rq_alpha_init   = parseFloat(document.getElementById(`${prefix}_rqAlphaInit`).value);
            config.rq_alpha_bounds = [
                parseFloat(document.getElementById(`${prefix}_rqAlphaLower`).value),
                parseFloat(document.getElementById(`${prefix}_rqAlphaUpper`).value)
            ];
        }
    }

    if (structTuning) {
        if (struct === "constant") {
            config.constant_value  = parseFloat(document.getElementById(`${prefix}_constVal`).value);
            config.constant_bounds = [
                parseFloat(document.getElementById(`${prefix}_constLower`).value),
                parseFloat(document.getElementById(`${prefix}_constUpper`).value)
            ];
        } else {
            config.noise_level = parseFloat(document.getElementById(`${prefix}_noiseLevel`).value);
            config.noise_lower = parseFloat(document.getElementById(`${prefix}_noiseLower`).value);
            config.noise_upper = parseFloat(document.getElementById(`${prefix}_noiseUpper`).value);
        }
    }
    return config;
}

/* ==============================
   RUN GPR
============================== */
async function runGPR(which) {
    const statusDiv = document.getElementById(`${which}GprStatus`);
    const timerDiv  = document.getElementById(`${which}GprTimer`);
    const errorDiv  = document.getElementById(`${which}GprError`);
    const btnId     = which === "std" ? "runStdBtn" : "runMeanBtn";
    const btn       = document.getElementById(btnId);

    // Clear old plots when re-running
    document.getElementById("plotArea").innerHTML = "";
    document.getElementById("plotSection").style.display = "none";
    if (which === "std") STD_DONE = false;
    else MEAN_DONE = false;

    errorDiv.innerHTML = "";
    statusDiv.style.display = "block";
    btn.disabled = true;

    let seconds = 0;
    const interval = setInterval(() => { timerDiv.innerHTML = `Elapsed: ${++seconds}s`; }, 1000);

    // Kernel bounds validation
    const prefix = which;
    if (document.getElementById(`${prefix}_baseTuning`)?.checked) {
        const lsLower = parseFloat(document.getElementById(`${prefix}_lsLower`).value);
        const lsUpper = parseFloat(document.getElementById(`${prefix}_lsUpper`).value);
        if (lsLower >= lsUpper) {
            clearInterval(interval);
            statusDiv.style.display = "none";
            btn.disabled = false;
            errorDiv.innerHTML = "Length scale lower bound must be less than upper bound.";
            return;
        }
        const kernelType = ["rbf","matern","rq"].find(t =>
            document.getElementById(`${prefix}_kernelBtn_${t}`)?.classList.contains("active")
        ) || "rbf";
        if (kernelType === "rq") {
            const rqLower = parseFloat(document.getElementById(`${prefix}_rqAlphaLower`).value);
            const rqUpper = parseFloat(document.getElementById(`${prefix}_rqAlphaUpper`).value);
            if (rqLower >= rqUpper) {
                clearInterval(interval);
                statusDiv.style.display = "none";
                btn.disabled = false;
                errorDiv.innerHTML = "RQ alpha lower bound must be less than upper bound.";
                return;
            }
        }
    }
    if (document.getElementById(`${prefix}_structTuning`)?.checked) {
        const struct = ["constant","white"].find(s =>
            document.getElementById(`${prefix}_structBtn_${s}`)?.classList.contains("active")
        ) || "constant";
        if (struct === "constant") {
            const cLower = parseFloat(document.getElementById(`${prefix}_constLower`).value);
            const cUpper = parseFloat(document.getElementById(`${prefix}_constUpper`).value);
            if (cLower >= cUpper) {
                clearInterval(interval);
                statusDiv.style.display = "none";
                btn.disabled = false;
                errorDiv.innerHTML = "Constant kernel lower bound must be less than upper bound.";
                return;
            }
        } else {
            const nLower = parseFloat(document.getElementById(`${prefix}_noiseLower`).value);
            const nUpper = parseFloat(document.getElementById(`${prefix}_noiseUpper`).value);
            if (nLower >= nUpper) {
                clearInterval(interval);
                statusDiv.style.display = "none";
                btn.disabled = false;
                errorDiv.innerHTML = "White noise lower bound must be less than upper bound.";
                return;
            }
        }
    }

    const payload = { mode: which };
    if (which === "std") {
        payload.std_num_cols     = STD_NUM_COLS;
        payload.std_cat_cols     = STD_CAT_COLS;
        payload.measurement_col  = STD_MEASUREMENT_COL;
        payload.std_gp_target    = STD_GP_TARGET;
        payload.std_log_vars     = STD_LOG_VARS;
        payload.std_noise        = document.getElementById("stdNoiseType").value;
        payload.std_kernel_config = getKernelConfig("std");
    } else {
        payload.mean_num_cols      = MEAN_NUM_COLS;
        payload.mean_cat_cols      = MEAN_CAT_COLS;
        payload.mean_output_col    = MEAN_OUTPUT_COL;
        payload.mean_log_vars      = MEAN_LOG_VARS;
        payload.mean_kernel_config = getKernelConfig("mean");
    }

    try {
        const res = await fetch("/run_gpr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        clearInterval(interval);
        statusDiv.style.display = "none";
        btn.disabled = false;

        const data = await res.json();
        if (data.error) { errorDiv.innerHTML = data.error; return; }

        const result = data[which];
        document.getElementById(`${which}Metrics`).innerHTML = `
            <div class="metrics-box">
                <strong>R²:</strong> ${result.r2} &nbsp;
                <strong>MAE:</strong> ${result.mae} &nbsp;
                <strong>RMSE:</strong> ${result.rmse}
            </div>`;

        if (which === "std") {
            STD_CONTROL_VARS    = result.control_vars;
            STD_CATEGORY_COMBOS = result.category_combos;
            STD_GP_TARGET       = result.gp_target;
            STD_DONE = true;
        } else {
            MEAN_CONTROL_VARS    = result.control_vars;
            MEAN_CATEGORY_COMBOS = result.category_combos;
            MEAN_DONE = true;
        }

        const shouldShow = ANALYSIS_MODE === "both" ? (STD_DONE && MEAN_DONE) : true;
        if (shouldShow) {
            populatePlotDropdowns();
            document.getElementById("plotSection").style.display = "block";
        }

    } catch (err) {
        clearInterval(interval);
        statusDiv.style.display = "none";
        btn.disabled = false;
        errorDiv.innerHTML = "Training failed.";
        console.error(err);
    }
}

/* ==============================
   PLOTTING
============================== */
function populatePlotDropdowns() {
    document.getElementById("unscaleStdRow").style.display = "block";

    const numCols = ANALYSIS_MODE === "mean" ? MEAN_NUM_COLS :
                    ANALYSIS_MODE === "std"  ? STD_NUM_COLS  :
                    [...new Set([...STD_NUM_COLS, ...MEAN_NUM_COLS])];

    const xSel = document.getElementById("xVarSelect");
    const ySel = document.getElementById("yVarSelect");
    xSel.innerHTML = "";
    ySel.innerHTML = "";

    numCols.forEach(col => {
        [xSel, ySel].forEach(sel => {
            const opt = document.createElement("option");
            opt.value = col; opt.textContent = col;
            sel.appendChild(opt);
        });
    });
}

function setColorScheme(scheme) {
    SELECTED_COLOR = scheme;
    document.querySelectorAll(".color-btn").forEach(b => b.classList.remove("active"));
    document.getElementById("colorBtn_" + scheme).classList.add("active");

    // Auto-regenerate if plots already visible
    if (LAST_PLOT_TYPE && document.getElementById("plotSection").style.display !== "none") {
        generatePlot(LAST_PLOT_TYPE);
    }
}

async function generatePlot(type) {
    LAST_PLOT_TYPE = type;
    const xVar = document.getElementById("xVarSelect").value;
    const yVar = type === "2d" ? document.getElementById("yVarSelect").value : null;

    document.getElementById("yVarRow").style.display = type === "2d" ? "block" : "none";
    if (!xVar) { alert("Select an X variable."); return; }

    // Show/hide unscale toggle:
    // Hide for 1D mean mode (always auto-unscaled)
    // Hide for 1D std/both mode when gp_target is replicate mean (auto-unscaled)
    // Always show for 2D
    const stdIsReplicateMean = STD_GP_TARGET === "mean";
    const hideFor1d = type === "1d" && (
        ANALYSIS_MODE === "mean" ||
        ((ANALYSIS_MODE === "std" || ANALYSIS_MODE === "both") && stdIsReplicateMean)
    );
    document.getElementById("unscaleStdRow").style.display = hideFor1d ? "none" : "block";

    const logVars = ANALYSIS_MODE === "mean" ? MEAN_LOG_VARS :
                    ANALYSIS_MODE === "std"  ? STD_LOG_VARS  :
                    [...new Set([...STD_LOG_VARS, ...MEAN_LOG_VARS])];

    try {
        const unscaleStd = hideFor1d || document.getElementById("unscaleStdCheck").checked;

        const res = await fetch("/generate_plot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mode: ANALYSIS_MODE,
                plot_type: type,
                x_var: xVar,
                y_var: yVar,
                log_vars: logVars,
                color_scheme: SELECTED_COLOR,
                unscale_std: unscaleStd
            })
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        const plotArea = document.getElementById("plotArea");
        plotArea.innerHTML = "";

        const is2d = data.plot_type === "2d";

        if (data.mode === "both") {
            plotArea.className = "plot-grid-both";
            data.pairs.forEach((pair, i) => {
                if (is2d) {
                    plotArea.innerHTML += `
                        <div class="plot-grid-2d">
                            <div class="plot-card">
                                <div class="plot-label">Std GP</div>
                                <img src="data:image/png;base64,${pair.std}" style="width:100%;">
                                <a href="data:image/png;base64,${pair.std}" download="std_plot_${i+1}.png"><button>Download</button></a>
                            </div>
                            <div class="plot-card">
                                <div class="plot-label">Mean GP</div>
                                <img src="data:image/png;base64,${pair.mean}" style="width:100%;">
                                <a href="data:image/png;base64,${pair.mean}" download="mean_plot_${i+1}.png"><button>Download</button></a>
                            </div>
                        </div>`;
                } else {
                    plotArea.innerHTML += `
                        <div class="plot-pair">
                            <div class="plot-card">
                                <div class="plot-label">Std GP</div>
                                <img src="data:image/png;base64,${pair.std}" style="max-width:100%;">
                                <a href="data:image/png;base64,${pair.std}" download="std_plot_${i+1}.png"><button>Download</button></a>
                            </div>
                            <div class="plot-card">
                                <div class="plot-label">Mean GP</div>
                                <img src="data:image/png;base64,${pair.mean}" style="max-width:100%;">
                                <a href="data:image/png;base64,${pair.mean}" download="mean_plot_${i+1}.png"><button>Download</button></a>
                            </div>
                        </div>`;
                }
            });
        } else {
            plotArea.className = is2d ? "plot-grid-2d" : "plot-grid";
            data.images.forEach((imgB64, i) => {
                plotArea.innerHTML += `
                    <div class="plot-card">
                        <img src="data:image/png;base64,${imgB64}" style="max-width:100%;">
                        <a href="data:image/png;base64,${imgB64}" download="gp_plot_${type}_${i+1}.png"><button>Download</button></a>
                    </div>`;
            });
        }
    } catch (err) {
        console.error(err);
        alert("Plot generation failed.");
    }
}

/* ==============================
   DRAG & DROP HELPERS
============================== */
function renderDraggableList(containerId, columns) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    columns.forEach(col => {
        const li = document.createElement("li");
        li.textContent = col;
        li.id = "col_" + containerId + "_" + col;
        li.draggable = true;
        container.appendChild(li);
    });
}

function getListItems(containerId) {
    return [...document.querySelectorAll(`#${containerId} li`)]
        .map(li => (li.dataset.col || li.textContent.split("(")[0].trim()));
}

document.addEventListener("dragstart", function(e) {
    if (e.target.tagName === "LI") {
        e.dataTransfer.setData("text/plain", e.target.id);
    }
});

// Delegated dragover/drop for ALL dropzones including dynamically created ones
document.addEventListener("dragover", function(e) {
    if (e.target.classList.contains("dropzone") || e.target.closest(".dropzone")) {
        e.preventDefault();
    }
});

document.addEventListener("drop", function(e) {
    const zone = e.target.classList.contains("dropzone")
        ? e.target
        : e.target.closest(".dropzone");
    if (!zone) return;
    e.preventDefault();
    const id = e.dataTransfer.getData("text/plain");
    const el = document.getElementById(id);
    if (el) zone.appendChild(el);
});

document.addEventListener("DOMContentLoaded", function() {
    showStep(1);
});
