let ALL_COLUMNS = [];
let CONTROL_COLS = [];
let METADATA_COLS = [];
let OUTPUT_COLS = [];
let FEATURE_EXPRESSIONS = [];

let CURRENT_MODE = null;
let MANUAL_NUM = [];
let MANUAL_CAT = [];

let TARGET_COL = null;

// Automatic mode state
let AUTO_NUM = [];
let AUTO_CAT = [];
let AUTO_DROPPED = [];

// Plot
let SELECTED_COLOR = "default";
let CATEGORY_SLICES = [];

let CURRENT_STEP = 1;

/* ==============================
   STEP WIZARD SYSTEM
============================== */

function showStep(stepNumber) {

    document.querySelectorAll(".step").forEach(step => {
        step.style.display = "none";
    });

    const stepMap = {
        1: "uploadStep",
        2: "columnStep",
        3: "featureEngineeringStep",
        4: "featureSelectionStep",
        5: "trainingStep"
    };

    const target = document.getElementById(stepMap[stepNumber]);
    if (target) {
        target.style.display = "block";
    }

    CURRENT_STEP = stepNumber;
}

function nextStep() {
    if (CURRENT_STEP < 5) {
        showStep(CURRENT_STEP + 1);
    }
}

function prevStep() {
    showStep(CURRENT_STEP - 1);
}

/* ==============================
   UPLOAD
============================== */

async function uploadFile() {
    const file = document.getElementById("fileInput").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    ALL_COLUMNS = data.columns;

    renderAllColumnSelectors();
    showStep(2);
}

function renderCheckboxList(containerId, columns) {

    const container = document.getElementById(containerId);
    container.innerHTML = "";

    columns.forEach(col => {

        const wrapper = document.createElement("label");

        wrapper.innerHTML = `
            <input type="checkbox" value="${col}">
            ${col}
        `;

        const checkbox = wrapper.querySelector("input");

        // Live filtering logic
        if (containerId === "controlColumns") {
            checkbox.addEventListener("change", updateMetadataOptions);
        }

        if (containerId === "metadataColumns") {
            checkbox.addEventListener("change", updateOutputOptions);
        }

        container.appendChild(wrapper);
        container.appendChild(document.createElement("br"));
    });
}

function getChecked(containerId) {
    return [...document.querySelectorAll(`#${containerId} input:checked`)]
        .map(x => x.value);
}

/* ==============================
   COLUMN SELECTION FLOW
============================== */

function renderAllColumnSelectors() {

    renderCheckboxList("controlColumns", ALL_COLUMNS);
    updateMetadataOptions();
    updateOutputOptions();
}

function updateMetadataOptions() {

    CONTROL_COLS = getChecked("controlColumns");

    const metadataOptions = ALL_COLUMNS.filter(
        col => !CONTROL_COLS.includes(col)
    );

    renderCheckboxList("metadataColumns", metadataOptions);

    updateOutputOptions();
}

function updateOutputOptions() {

    CONTROL_COLS = getChecked("controlColumns");
    METADATA_COLS = getChecked("metadataColumns");

    const outputOptions = ALL_COLUMNS.filter(
        col => !CONTROL_COLS.includes(col) &&
               !METADATA_COLS.includes(col)
    );

    renderCheckboxList("outputColumns", outputOptions);
}

function confirmColumnSelection() {

    CONTROL_COLS = getChecked("controlColumns");
    METADATA_COLS = getChecked("metadataColumns");
    OUTPUT_COLS = getChecked("outputColumns");

    if (OUTPUT_COLS.length === 0) {
        alert("Please select at least one output column.");
        return;
    }

    TARGET_COL = OUTPUT_COLS[0];

    // Clear UI visually (state remains stored)
    document.getElementById("controlColumns").innerHTML = "";
    document.getElementById("metadataColumns").innerHTML = "";
    document.getElementById("outputColumns").innerHTML = "";

    nextStep();
}

async function runDetection() {

    OUTPUT_COLS = getChecked("outputColumns");

    if (OUTPUT_COLS.length === 0) {
        alert("Please select at least one output column.");
        return;
    }

    TARGET_COL = OUTPUT_COLS[0];

    try {
        const res = await fetch("/detect", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                control_cols: CONTROL_COLS,
                metadata_cols: METADATA_COLS,
                output_cols: OUTPUT_COLS
            })
        });

        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        if (data.outliers_detected) {
            alert("Outliers detected! You may proceed to GPR.");
            nextStep();
        } else {
            alert("No outliers detected. GPR disabled.");
        }

    } catch (err) {
        console.error(err);
        alert("Detection failed.");
    }
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
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ expression: value })
        });

        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        ALL_COLUMNS = data.columns;

        const list = document.getElementById("featureList");
        const li = document.createElement("li");
        li.textContent = value;
        list.appendChild(li);

        input.value = "";
        alert("Feature added.");

    } catch (err) {
        console.error(err);
        alert("Feature failed.");
    }
}

/* ==============================
   MODE SELECTION
============================== */

function selectMode(mode) {
    CURRENT_MODE = mode;

    document.getElementById("featureSelectionArea").style.display = "block";
    document.getElementById("manualSection").style.display = "none";
    document.getElementById("autoSection").style.display = "none";
    document.getElementById("runGprContainer").style.display = "none";

    if (mode === "manual") {
        initManual();
        document.getElementById("manualSection").style.display = "block";
    } else {
        initAuto();
        document.getElementById("autoSection").style.display = "block";
    }
}

/* ==============================
   MANUAL MODE (Drag & Drop)
============================== */

function initManual() {

    if (!TARGET_COL) {
        alert("Run detection first.");
        return;
    }

    const available = ALL_COLUMNS.filter(col => col !== TARGET_COL);

    renderDraggable("availableColumns", available);

    document.getElementById("manualNum").innerHTML = "";
    document.getElementById("manualCat").innerHTML = "";
}

function confirmManual() {
    MANUAL_NUM = getListItems("manualNum");
    MANUAL_CAT = getListItems("manualCat");

    if (MANUAL_NUM.length === 0 && MANUAL_CAT.length === 0) {
        alert("Select at least one feature.");
        return;
    }

    showStep(5);
}

function renderDraggable(containerId, columns) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    columns.forEach(col => {
        const li = document.createElement("li");
        li.textContent = col;
        li.id = "col_" + col;
        li.draggable = true;
        container.appendChild(li);
    });
}

function getListItems(containerId) {
    return [...document.querySelectorAll(`#${containerId} li`)]
        .map(li => li.textContent);
}

document.addEventListener("dragstart", function(e) {
    if (e.target.tagName === "LI") {
        e.dataTransfer.setData("text/plain", e.target.id);
    }
});

document.querySelectorAll(".dropzone").forEach(zone => {

    zone.addEventListener("dragover", e => e.preventDefault());

    zone.addEventListener("drop", function(e) {
        e.preventDefault();
        const id = e.dataTransfer.getData("text/plain");
        const element = document.getElementById(id);
        if (element) this.appendChild(element);
    });

});

/* ==============================
   AUTOMATIC MODE (X + Restore)
============================== */

async function initAuto() {

    if (!TARGET_COL) {
        alert("Run detection first.");
        return;
    }

    const res = await fetch("/auto_detect_features", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ target_col: TARGET_COL })
    });

    const data = await res.json();

    AUTO_NUM = [...data.num_cols];
    AUTO_CAT = [...data.cat_cols];
    AUTO_DROPPED = [];

    renderAutoLists();
}

function renderAutoLists() {

    const numList = document.getElementById("autoNum");
    const catList = document.getElementById("autoCat");
    const dropList = document.getElementById("autoDropped");

    numList.innerHTML = "";
    catList.innerHTML = "";
    dropList.innerHTML = "";

    AUTO_NUM.forEach(col => {
        numList.innerHTML += `
            <li>
                ${col}
                <button class="feature-btn" onclick="dropFeature('${col}', 'num')">❌</button>
            </li>`;
    });

    AUTO_CAT.forEach(col => {
        catList.innerHTML += `
            <li>
                ${col}
                <button class="feature-btn" onclick="dropFeature('${col}', 'cat')">❌</button>
            </li>`;
    });

    AUTO_DROPPED.forEach(obj => {
        dropList.innerHTML += `
            <li>
                ${obj.name}
                <button class="feature-btn restore-btn" onclick="restoreFeature('${obj.name}', '${obj.type}')">➕</button>
            </li>`;
    });
}

function dropFeature(col, type) {

    if (type === "num") {
        AUTO_NUM = AUTO_NUM.filter(c => c !== col);
    } else {
        AUTO_CAT = AUTO_CAT.filter(c => c !== col);
    }

    AUTO_DROPPED.push({ name: col, type: type });

    renderAutoLists();
}

function restoreFeature(col, type) {

    AUTO_DROPPED = AUTO_DROPPED.filter(obj => obj.name !== col);

    if (type === "num") AUTO_NUM.push(col);
    else AUTO_CAT.push(col);

    renderAutoLists();
}

function confirmAuto() {

    if (AUTO_NUM.length === 0 && AUTO_CAT.length === 0) {
        alert("At least one feature must remain.");
        return;
    }

    showStep(5);
}

/* ==============================
   RUN GPR + TIMER
============================== */

async function runGPR() {

    let payload = {
        target_col: TARGET_COL,
        mode: CURRENT_MODE,
        num_cols: [],
        cat_cols: [],
        kernel_config: {}
    };

    if (CURRENT_MODE === "manual") {
        payload.num_cols = [...MANUAL_NUM];
        payload.cat_cols = [...MANUAL_CAT];
    }

    if (CURRENT_MODE === "auto") {
        payload.num_cols = [...AUTO_NUM];
        payload.cat_cols = [...AUTO_CAT];
            }
    const advToggle = document.getElementById("advancedModeToggle");
    const advanced = advToggle ? advToggle.checked : false;

    payload.kernel_config = {
        kernel_type: document.getElementById("kernelType").value,
        advanced: advanced
    };
    if (advanced) {
        payload.kernel_config.length_scale_init =
            parseFloat(document.getElementById("lengthScaleInit").value);

        payload.kernel_config.length_scale_lower =
            parseFloat(document.getElementById("lengthScaleLower").value);

        payload.kernel_config.length_scale_upper =
            parseFloat(document.getElementById("lengthScaleUpper").value);

        payload.kernel_config.noise_level =
            parseFloat(document.getElementById("noiseLevel").value);

        payload.kernel_config.alpha =
            parseFloat(document.getElementById("alphaValue").value);

        payload.kernel_config.restarts =
            parseInt(document.getElementById("restartCount").value);
    }
    const statusDiv = document.getElementById("gprStatus");
    const timerDiv = document.getElementById("gprTimer");
    const errorDiv = document.getElementById("gprError");
    const button = document.getElementById("runGprBtn");

    errorDiv.innerHTML = "";
    statusDiv.style.display = "block";
    button.disabled = true;

    let seconds = 0;
    const interval = setInterval(() => {
        seconds++;
        timerDiv.innerHTML = `Elapsed: ${seconds}s`;
    }, 1000);

    try {
        const res = await fetch("/run_gpr", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        clearInterval(interval);
        statusDiv.style.display = "none";
        button.disabled = false;

        const data = await res.json();

        if (data.error) {
            errorDiv.innerHTML = data.error;
            return;
        }

        document.getElementById("metrics").innerHTML =
            `<h4>Performance</h4>
            R²: ${data.r2}<br>
            MAE: ${data.mae}<br>
            RMSE: ${data.rmse}`;

        document.getElementById("modelSummary").innerHTML =
            `<h4>Model Summary</h4>
            Features Used: ${data.feature_count}<br>
            Train Size: ${data.train_size}<br>
            Test Size: ${data.test_size}`;
        document.getElementById("plotPrompt").style.display = "block";

        if (data.feature_importance) {

            let importanceHTML = "<h4>Feature Importance (Numeric)</h4><ol>";

            data.feature_importance.forEach(item => {
                importanceHTML += `
                    <li>
                        ${item.feature} 
                        <span style="opacity:0.6;">(${item.importance.toFixed(4)})</span>
                    </li>
                `;
            });

            importanceHTML += "</ol>";

            document.getElementById("modelSummary").innerHTML += importanceHTML;
        }

    } catch (err) {
        clearInterval(interval);
        statusDiv.style.display = "none";
        button.disabled = false;
        errorDiv.innerHTML = "Training failed.";
        console.error(err);
    }
}
/* ==============================
   PLOTTING (Sequential Category Flow)
============================== */

let CURRENT_SLICE_INDEX = 0;

function showColorOptions() {
    document.getElementById("colorOptions").style.display = "block";
}

function skipPlots() {
    document.getElementById("plotPrompt").style.display = "none";
}

async function setColorScheme(scheme) {

    SELECTED_COLOR = scheme;

    const res = await fetch("/get_category_slices", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
    });

    const data = await res.json();

    CATEGORY_SLICES = data.slices;
    CURRENT_SLICE_INDEX = 0;

    document.getElementById("plotControls").style.display = "block";
    document.getElementById("plotArea").innerHTML = "";

    showCurrentCategory();
}

function showCurrentCategory() {

    const slice = CATEGORY_SLICES[CURRENT_SLICE_INDEX];

    const container = document.getElementById("plotControls");

    container.innerHTML = `
        <h4>Category Slice ${CURRENT_SLICE_INDEX + 1} of ${CATEGORY_SLICES.length}</h4>
        <div><strong>${JSON.stringify(slice)}</strong></div>
        <br>
        <button onclick="generatePlot('1d')">Plot 1D</button>
        <button onclick="generatePlot('2d')">Plot 2D</button>
        <button onclick="generatePlot('3d')">Plot 3D</button>
        <br><br>
        <br><br>
        <button onclick="nextCategory()">Next Category →</button>
    `;

    document.getElementById("plotArea").innerHTML = "";
}

async function generatePlot(type) {

    const slice = CATEGORY_SLICES[CURRENT_SLICE_INDEX];

    const res = await fetch("/generate_plot", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            slice: slice,
            plot_type: type,
            color_scheme: SELECTED_COLOR,
            target_col: TARGET_COL
        })
    });

    const data = await res.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    const plotArea = document.getElementById("plotArea");

    // 🔥 Clear previous plots (prevents duplicates)
    plotArea.innerHTML = "";

    // 🔥 Support multiple images (1D = many features, 2D = many pairs)
    if (data.images && Array.isArray(data.images)) {

        data.images.forEach((imgBase64, index) => {

            plotArea.innerHTML += `
                <div style="margin-bottom:20px;">
                    <img src="data:image/png;base64,${imgBase64}" />
                    <br>
                    <a href="data:image/png;base64,${imgBase64}" 
                       download="gp_plot_slice_${CURRENT_SLICE_INDEX + 1}_${index + 1}.png">
                        <button>Download Plot</button>
                    </a>
                </div>
            `;
        });

    } 
    // 🔁 Backward compatibility (single image case)
    else if (data.image) {

        plotArea.innerHTML = `
            <div style="margin-bottom:20px;">
                <img src="data:image/png;base64,${data.image}" />
                <br>
                <a href="data:image/png;base64,${data.image}" 
                   download="gp_plot_slice_${CURRENT_SLICE_INDEX + 1}.png">
                    <button>Download Plot</button>
                </a>
            </div>
        `;
    }
}

function nextCategory() {

    CURRENT_SLICE_INDEX++;

    if (CURRENT_SLICE_INDEX >= CATEGORY_SLICES.length) {
        alert("All categories completed.");
        document.getElementById("plotControls").innerHTML = "<h4>All categories processed.</h4>";
        document.getElementById("plotArea").innerHTML = "";
        return;
    }

    showCurrentCategory();
}

document.addEventListener("DOMContentLoaded", function () {

    // Step initialization
    showStep(1);

    // Advanced mode toggle (safe guard)
    const advToggle = document.getElementById("advancedModeToggle");

    if (advToggle) {
        advToggle.addEventListener("change", function() {
            const adv = document.getElementById("kernelAdvancedSection");
            if (adv) {
                adv.style.display = this.checked ? "block" : "none";
            }
        });
    }

});
