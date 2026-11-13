let modelWorker;
let classesPromise = fetch("./model/classes.json").then(response => response.json());

async function browserPredict() {
  const image = new Image();
  image.src = state.source;
  await image.decode();
  const scale = Math.min(640 / image.naturalWidth, 640 / image.naturalHeight);
  const width = Math.round(image.naturalWidth * scale), height = Math.round(image.naturalHeight * scale);
  const x = Math.floor((640 - width) / 2), y = Math.floor((640 - height) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 640;
  const context = canvas.getContext("2d");
  context.fillStyle = "rgb(124,116,104)";
  context.fillRect(0,0,640,640);
  context.drawImage(image,x,y,width,height);
  const rgba = context.getImageData(0,0,640,640).data;
  const pixels = new Float32Array(3*640*640);
  for (let i=0; i<640*640; i++) {
    pixels[i]=rgba[i*4]/255;
    pixels[i+640*640]=rgba[i*4+1]/255;
    pixels[i+2*640*640]=rgba[i*4+2]/255;
  }
  const names = await classesPromise;
  if (!modelWorker) modelWorker = new Worker("./inference-worker.js");
  const output = await new Promise((resolve,reject) => {
    modelWorker.onmessage = ({data}) => {
      if (data.status) $("trayStatus").textContent = data.status;
      if (data.result) resolve(data.result);
      if (data.error) reject(new Error(data.error));
    };
    modelWorker.onerror = () => { modelWorker.terminate(); modelWorker=null; reject(new Error("Browser inference could not start. Please try a desktop browser.")); };
    modelWorker.postMessage({pixels},[pixels.buffer]);
  });
  const clip = (value,max) => Math.min(max,Math.max(0,value));
  return {detections:output.scores.map((score,i) => ({
    score, class_id:output.labels[i], label:names[output.labels[i]],
    box:[clip((output.boxes[4*i]-x)*image.naturalWidth/width,image.naturalWidth),
         clip((output.boxes[4*i+1]-y)*image.naturalHeight/height,image.naturalHeight),
         clip((output.boxes[4*i+2]-x)*image.naturalWidth/width,image.naturalWidth),
         clip((output.boxes[4*i+3]-y)*image.naturalHeight/height,image.naturalHeight)]
  })).filter(item=>item.box[2]>item.box[0] && item.box[3]>item.box[1])};
}
const scenes = [
  { name: "VOC scene 000001", hint: "Person in scene", image: "./examples/000001.png", result: "./examples/000001.json" },
  { name: "VOC scene 000003", hint: "Indoor scene", image: "./examples/000003.png", result: "./examples/000003.json" }
];
const $ = id => document.getElementById(id);
const state = {source:null, name:"", file:null, scene:null, result:null, busy:false};
const list = $("sceneList"), stages = [...document.querySelectorAll(".stage")], connectors = [...document.querySelectorAll(".connector")];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const setStatus = (text) => { $("flowStatus").textContent = text; };
const stageDescriptions = [
  "ResNet-50 and the feature pyramid extract patterns at several scales.",
  "The region proposal network finds candidate boxes that may contain objects.",
  "RoI pooling collects a fixed-size feature representation for each region.",
  "The fine-tuned predictor assigns a class and refines each box."
];
const detail = document.createElement("p");
detail.className = "stage-detail";
detail.textContent = "Explore a stage to see what it does.";
$("pipeline").after(detail);
stages.forEach((stage,index) => {
  stage.tabIndex = 0;
  stage.setAttribute("role","button");
  stage.setAttribute("aria-label",stageDescriptions[index]);
  const explain = () => { detail.textContent=stageDescriptions[index]; };
  stage.addEventListener("click",explain);
  stage.addEventListener("keydown",event => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); explain(); }
  });
});

function flyToInput(image) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const from = image.getBoundingClientRect(), to = $("inputPort").getBoundingClientRect();
  const clone = image.cloneNode();
  Object.assign(clone.style,{position:"fixed",left:from.left+"px",top:from.top+"px",width:from.width+"px",height:from.height+"px",objectFit:"cover",zIndex:1000,borderRadius:"8px",pointerEvents:"none"});
  document.body.append(clone);
  const animation = clone.animate([
    {transform:"translate(0,0) rotate(-4deg)",opacity:1},
    {transform:`translate(${to.left-from.left}px,${to.top-from.top}px) scale(${to.width/from.width},${to.height/from.height})`,transformOrigin:"top left",opacity:0}
  ],{duration:650,easing:"cubic-bezier(.22,.8,.3,1)"});
  animation.onfinish = () => clone.remove();
}

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
  loadInput();
  flyToInput(document.querySelectorAll(".scene")[index].querySelector("img"));
}

$("fileInput").addEventListener("change", event => {
  const file = event.target.files[0];
  if (file) useFile(file);
});
function useFile(file) {
  if (state.busy) return;
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
  loadInput();
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
  port.animate([{transform:"translateY(-12px) scale(.85)",opacity:.3},{transform:"translateY(0) scale(1)",opacity:1}],{duration:500,easing:"cubic-bezier(.2,.8,.2,1)"});
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
  $("resultNote").textContent = "Uploads run on your device. First run downloads 166 MB of trained model weights.";
  stages.forEach(stage => stage.classList.remove("active","done"));
  connectors.forEach(line => line.classList.remove("lit"));
}
async function sendToModel() {
  if (state.scene) {
    const response = await fetch(state.scene.result);
    if (!response.ok) throw new Error("Saved prediction could not be loaded.");
    return {...await response.json(),savedExample:true};
  }
  return browserPredict();
}
async function run() {
  if (!state.source || state.busy) return;
  state.busy = true;
  $("runButton").disabled = true;
  loadInput();
  clearResult();
  $("resetButton").disabled = true;
  document.querySelector(".workspace").classList.add("running");
  const prediction = sendToModel().catch(error => ({error:error.message}));
  for (let i = 0; i < stages.length; i++) {
    setStatus(["EXTRACTING FEATURES","FINDING REGIONS","POOLING REGIONS","APPLYING WEIGHTS"][i]);
    stages[i].classList.add("active");
    detail.textContent = stageDescriptions[i];
    await pause(650);
    stages[i].classList.remove("active");
    stages[i].classList.add("done");
    if (connectors[i]) connectors[i].classList.add("lit");
  }
  const waiting = setInterval(() => {
    stages.forEach(stage => stage.classList.toggle("active"));
  }, 850);
  const response = await prediction;
  clearInterval(waiting);
  stages.forEach(stage => stage.classList.remove("active"));
  document.querySelector(".workspace").classList.remove("running");
  $("resetButton").disabled = false;
  if (response?.detections) {
    state.result = response;
    $("resultMode").textContent = response.savedExample ? "SAVED MODEL RESULT" : "LIVE MODEL RESULT";
    $("resultNote").textContent = response.savedExample ? "Real detections saved by the trained model for this VOC image. Upload your own photo to run the trained model in your browser." : "Boxes and confidence scores returned by the trained Faster R-CNN model.";
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
  setStatus(response?.error ? "PLEASE RETRY" : "COMPLETE");
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
  if (!found.length && !state.result.error) $("detectionList").textContent = "No detections above this threshold.";
}
$("runButton").addEventListener("click", run);
$("threshold").addEventListener("input", renderBoxes);
window.addEventListener("resize", renderBoxes);
$("resetButton").addEventListener("click", () => {
  if (state.busy) return;
  state.source = null; state.file = null; state.scene = null; state.result = null;
  $("fileInput").value = ""; $("inputImage").hidden = true; port.classList.remove("loaded");
  $("inputLabel").textContent = "Drop an image here"; $("trayStatus").textContent = "Select a scene to begin.";
  $("runButton").disabled = true; setStatus("READY");
  document.querySelectorAll(".scene").forEach(button => button.classList.remove("selected"));
  clearResult();
});
