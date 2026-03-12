let ALL_COLUMNS = []

let CONTROL_VARS = []
let BLOCK_VAR = null
let OUTPUT_VAR = null
let LOG_VARS = []
let OUTLIER_SD = 1

let CURRENT_BLOCK = null
let CURRENT_STEP = 1


function showStep(n) {

    document.querySelectorAll(".step").forEach(step => {
        step.style.display = "none"
    })

    const stepMap = {
        1: "uploadStep",
        2: "columnStep",
        3: "trainingStep"
    }

    document.getElementById(stepMap[n]).style.display = "block"

    CURRENT_STEP = n
}

function nextStep() {
    showStep(CURRENT_STEP + 1)
}

function prevStep() {
    showStep(CURRENT_STEP - 1)
}


/* UPLOAD */

async function uploadFile() {

    const file = document.getElementById("fileInput").files[0]

    const form = new FormData()
    form.append("file", file)

    const res = await fetch("/upload", {
        method: "POST",
        body: form
    })

    const data = await res.json()

    ALL_COLUMNS = data.columns

    renderColumns()

    showStep(2)
}


/* COLUMN RENDER */

function renderColumns() {

    renderCheckbox("controlColumns", ALL_COLUMNS)
    renderRadio("blockColumn", ALL_COLUMNS)
    renderRadio("outputColumn", ALL_COLUMNS)
    renderCheckbox("logColumns", ALL_COLUMNS)
}

function renderCheckbox(containerId, columns) {

    const container = document.getElementById(containerId)

    container.innerHTML = ""

    columns.forEach(col => {

        container.innerHTML += `
            <label>
                <input type="checkbox" value="${col}">
                ${col}
            </label><br>
        `
    })
}

function renderRadio(containerId, columns) {

    const container = document.getElementById(containerId)

    container.innerHTML = ""

    columns.forEach(col => {

        container.innerHTML += `
            <label>
                <input type="radio" name="${containerId}" value="${col}">
                ${col}
            </label><br>
        `
    })
}

function getChecked(containerId) {

    return [...document.querySelectorAll(`#${containerId} input:checked`)]
        .map(x => x.value)
}

function getRadio(containerId) {

    const el = document.querySelector(`#${containerId} input:checked`)

    return el ? el.value : null
}


/* CONFIRM COLUMNS */

function confirmColumns() {

    CONTROL_VARS = getChecked("controlColumns")
    BLOCK_VAR = getRadio("blockColumn")
    OUTPUT_VAR = getRadio("outputColumn")
    LOG_VARS = getChecked("logColumns")

    OUTLIER_SD = parseFloat(document.getElementById("outlierSD").value)

    loadBlocks()

    showStep(3)
}


/* GROUP DATA */

async function loadBlocks() {

    const res = await fetch("/group", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

            control_vars: CONTROL_VARS,
            block_var: BLOCK_VAR,
            output_var: OUTPUT_VAR,
            log_vars: LOG_VARS,
            outlier_sd: OUTLIER_SD

        })
    })

    const data = await res.json()

    renderBlocks(data.blocks)
}

function renderBlocks(blocks) {

    const container = document.getElementById("blockSelection")

    container.innerHTML = ""

    blocks.forEach(block => {

        const btn = document.createElement("button")

        btn.innerText = "Train GP for " + block

        btn.onclick = () => trainBlock(block)

        container.appendChild(btn)
    })
}


/* TRAIN GP */

async function trainBlock(block) {

    CURRENT_BLOCK = block

    const kernel = document.getElementById("kernelType").value

    const res = await fetch("/run_gp", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

            block: block,
            control_vars: CONTROL_VARS,
            log_vars: LOG_VARS,
            kernel_type: kernel

        })
    })

    const data = await res.json()

    alert("GP trained for " + block)
}


/* PLOT */

async function generatePlot(type) {

    const res = await fetch("/generate_plot", {

        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({

            dimension: type,
            xVar: CONTROL_VARS[0],
            yVar: CONTROL_VARS[1]

        })
    })

    const data = await res.json()

    document.getElementById("plotArea").innerHTML =
        `<img src="data:image/png;base64,${data.image}" />`
}


document.addEventListener("DOMContentLoaded", function () {

    showStep(1)

    document.getElementById("advancedModeToggle")
        .addEventListener("change", function () {

            document.getElementById("kernelAdvancedSection").style.display =
                this.checked ? "block" : "none"

        })
})