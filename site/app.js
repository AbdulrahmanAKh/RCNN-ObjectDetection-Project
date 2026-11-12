const API_URL = window.DETECTOR_API_URL || "";
const scenes = [
  { name: "VOC scene 000001", hint: "Person in scene", image: "./examples/000001.png", result: "./examples/000001.json" },
  { name: "VOC scene 000003", hint: "Indoor scene", image: "./examples/000003.png", result: "./examples/000003.json" }
];
const $ = id => document.getElementById(id);
const state = {source:null, name:"", file:null, scene:null, result:null, busy:false};
const list = $("sceneList"), stages = [...document.querySelectorAll(".stage")], connectors = [...document.querySelectorAll(".connector")];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const setStatus = (text) => { $("flowStatus").textContent = text; };

scenes.forEach((scene, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "scene";
  button.draggable = true;
  button.innerHTML = `<img src="${scene.image}" alt=""><span><strong>${scene.name}</strong><small>${scene.hint}</small></span>`;
  button.addEventListener("click", () => selectScene(index));
  button.addEventListener("dragstart", event => {
    selectScene(index);
    event.dataTransfer.setData("text/plain", String(index));
    event.dataTransfer.effectAllowed = "copy";
  });
  list.append(button);
});

function selectScene(index) {
  if (state.busy) return;
  const scene = scenes[index];
  state.source = scene.image;
  state.name = scene.name;
  state.file = null;
  state.scene = scene;
  state.result = null;
  document.querySelectorAll(".scene").forEach((button, i) => button.classList.toggle("selected", i === index));
  $("trayStatus").textContent = `${scene.name} selected. Drag it to the input port or press Run detection.`;
  $("runButton").disabled = false;
  clearResult();
}

$("fileInput").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) useFile(file);
});
function useFile(file) {
  if (!["image/jpeg","image/png","image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024) {
    $("trayStatus").textContent = "Choose a JPG, PNG, or WebP image under 10 MB.";
    return;
  }
  if (state.source?.startsWith("blob:")) URL.revokeObjectURL(state.source);
  state.source = URL.createObjectURL(file);
  state.name = file.name;
  state.file = file;
  state.scene = null;
  state.result = null;
  document.querySelectorAll(".scene").forEach(button => button.classList.remove("selected"));
  $("trayStatus").textContent = `${file.name} is ready.`;
  $("runButton").disabled = false;
  clearResult();
}

const port = $("inputPort");
port.addEventListener("dragover", event => { event.preventDefault(); port.classList.add("dragover"); });
port.addEventListener("dragleave", () => port.classList.remove("dragover"));
port.addEventListener("drop", event => {
  event.preventDefault();
  port.classList.remove("dragover");
  if (event.dataTransfer.files.length) useFile(event.dataTransfer.files[0]);
  else if (event.dataTransfer.getData("text/plain")) selectScene(Number(event.dataTransfer.getData("text/plain")));
  loadInput();
});
port.addEventListener("click", () => state.source ? loadInput() : $("fileInput").click());
port.addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); port.click(); }
});
function loadInput() {
  if (!state.source) return;
  $("inputImage").src = state.source;
  $("inputImage").hidden = false;
  port.classList.add("loaded");
  $("inputLabel").textContent = state.name;
  setStatus("IMAGE LOADED");
}
function clearResult() {
  $("resultImage").hidden = true;
  $("resultImage").removeAttribute("src");
  $("boxes").replaceChildren();
  $("detectionList").replaceChildren();
  $("emptyResult").hidden = false;
  $("resultMode").textContent = "WAITING";
  $("resultNote").textContent = "The animation explains the model stages. Detection results appear when the inference service is connected.";
  stages.forEach(stage => stage.classList.remove("active","done"));
  connectors.forEach(line => line.classList.remove("lit"));
}
async function sendToModel() {
  if (!API_URL) {
    if (state.scene) {
      const response = await fetch(state.scene.result);
      if (!response.ok) throw new Error("Saved prediction could not be loaded.");
      return {...await response.json(), savedExample:true};
    }
    return null;
  }
  const form = new FormData();
  if (state.file) form.append("image", state.file);
  else {
    const response = await fetch(state.source);
    form.append("image", await response.blob(), "sample.jpg");
  }
  const response = await fetch(API_URL.replace(/\/$/,"") + "/predict", {method:"POST",body:form});
  if (!response.ok) throw new Error(`Inference service returned ${response.status}`);
  return response.json();
}
async function run() {
  if (!state.source || state.busy) return;
  state.busy = true;
  $("runButton").disabled = true;
  loadInput();
  clearResult();
  const prediction = sendToModel().catch(error => ({error:error.message}));
  for (let i = 0; i < stages.length; i++) {
    setStatus(["EXTRACTING FEATURES","FINDING REGIONS","POOLING REGIONS","APPLYING WEIGHTS"][i]);
    stages[i].classList.add("active");
    await pause(650);
    stages[i].classList.remove("active");
    stages[i].classList.add("done");
    if (connectors[i]) connectors[i].classList.add("lit");
  }
  const response = await prediction;
  if (response?.detections) {
    state.result = response;
    $("resultMode").textContent = response.savedExample ? "SAVED MODEL RESULT" : "LIVE MODEL RESULT";
    $("resultNote").textContent = response.savedExample ? "Real detections saved by the trained model for this VOC image. Uploads need the inference service." : "Boxes and confidence scores returned by the trained Faster R-CNN model.";
  } else {
    state.result = {detections:[]};
    $("resultMode").textContent = "INFERENCE UNAVAILABLE";
    $("resultNote").textContent = response?.error || "Your photo was loaded. Connect the trained model service to obtain detections.";
  }
  $("resultImage").src = state.source;
  $("resultImage").hidden = false;
  $("emptyResult").hidden = true;
  $("resultImage").onload = renderBoxes;
  renderBoxes();
  setStatus("COMPLETE");
  state.busy = false;
  $("runButton").disabled = false;
}
function renderBoxes() {
  const image = $("resultImage");
  if (!image.complete || !image.naturalWidth || !state.result) return;
  const area = $("resultVisual").getBoundingClientRect();
  const scale = Math.min(area.width / image.naturalWidth, area.height / image.naturalHeight);
  const width = image.naturalWidth * scale, height = image.naturalHeight * scale;
  const left = (area.width - width) / 2, top = (area.height - height) / 2;
  const limit = Number($("threshold").value);
  $("thresholdValue").value = limit.toFixed(2);
  $("boxes").replaceChildren();
  $("detectionList").replaceChildren();
  const found = state.result.detections.filter(item => item.score >= limit);
  for (const item of found) {
    const coords = [item.box[0]/image.naturalWidth,item.box[1]/image.naturalHeight,item.box[2]/image.naturalWidth,item.box[3]/image.naturalHeight];
    const box = document.createElement("div");
    box.className = "box";
    box.style.left = `${left + coords[0]*width}px`;
    box.style.top = `${top + coords[1]*height}px`;
    box.style.width = `${(coords[2]-coords[0])*width}px`;
    box.style.height = `${(coords[3]-coords[1])*height}px`;
    const tag = document.createElement("span");
    tag.textContent = `${item.label} ${Math.round(item.score*100)}%`;
    box.append(tag);
    $("boxes").append(box);
    const row = document.createElement("div");
    row.className = "detection";
    row.innerHTML = `<span></span><b></b>`;
    row.children[0].textContent = item.label;
    row.children[1].textContent = `${Math.round(item.score*100)}%`;
    $("detectionList").append(row);
  }
  if (!found.length && $("resultMode").textContent === "MODEL RESULT") $("detectionList").textContent = "No detections above this threshold.";
}
$("runButton").addEventListener("click", run);
$("threshold").addEventListener("input", renderBoxes);
window.addEventListener("resize", renderBoxes);
$("resetButton").addEventListener("click", () => {
  state.source = null; state.file = null; state.scene = null; state.result = null;
  $("fileInput").value = ""; $("inputImage").hidden = true; port.classList.remove("loaded");
  $("inputLabel").textContent = "Drop an image here"; $("trayStatus").textContent = "Select a scene to begin.";
  $("runButton").disabled = true; setStatus("READY");
  document.querySelectorAll(".scene").forEach(button => button.classList.remove("selected"));
  clearResult();
});
