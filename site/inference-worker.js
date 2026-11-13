importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.min.js");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
let session;
async function loadModel() {
  if (session) return session;
  const parts = [];
  for (let i = 0; i < 3; i++) {
    postMessage({status: "Downloading model " + (i + 1) + "/3 · first run only"});
    const response = await fetch("./model/part-" + i + ".bin");
    if (!response.ok) throw new Error("Model download failed. Please refresh and retry.");
    parts.push(new Uint8Array(await response.arrayBuffer()));
  }
  const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.length; }
  postMessage({status:"Preparing trained model…"});
  session = await ort.InferenceSession.create(bytes, {executionProviders:["wasm"]});
  return session;
}
self.onmessage = async ({data}) => {
  try {
    const model = await loadModel();
    postMessage({status:"Detecting objects on your device…"});
    const result = await model.run({image:new ort.Tensor("float32", data.pixels, [1,3,640,640])});
    postMessage({result:{boxes:Array.from(result.boxes.data), labels:Array.from(result.labels.data, Number), scores:Array.from(result.scores.data)}});
  } catch (error) {
    postMessage({error:error.message});
  }
};
