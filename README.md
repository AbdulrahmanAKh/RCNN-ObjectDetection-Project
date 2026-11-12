# Inside the Detector

An interactive Faster R-CNN exhibit and the executed notebook that trains its PASCAL VOC 2007 model.

The attached full run selected epoch 10 with 81.7% validation mAP and reported 81.8% test mAP on 4,952 VOC 2007 test images. Its warm median inference time was 24 ms on an NVIDIA A100-SXM4-40GB. These values come from the notebook output and have not been measured on the eventual demo host.

## Run the website

Serve the repository root, then open `/site/`. For example, `python -m http.server 8000` and visit `http://localhost:8000/site/`.

The two bundled VOC scenes show **real saved detections** from the trained checkpoint. The confidence slider filters their saved results. Uploaded images require the inference service. The animated stages explain the model's architecture and do not display live internal activations.

## Train and connect the detector

1. Run `notebooks/FasterRCNN_VOC2007_Portfolio.ipynb` in Colab with a GPU and `MODE = "full"`.
2. The exported `voc-detector-full.zip` contains `best.pt`, `inference-requirements.txt`, metadata, and example outputs. Keep the model file on the inference host. The site's example images and saved results already come from this ZIP.
3. Install matching PyTorch and TorchVision versions for your host, plus `backend/requirements.txt`.
4. Start the backend with `MODEL_PATH=/path/to/best.pt SITE_ORIGINS=https://YOURNAME.github.io uvicorn app:app --host 0.0.0.0 --port 8001` from the `backend/` directory.
5. Put the backend's public HTTPS origin in `site/config.js`. GitHub Pages serves the frontend. The backend needs separate Python hosting because the browser cannot execute the PyTorch checkpoint.

The inference response uses original image pixel coordinates and includes `box`, `label`, and `score` for each detection. The confidence slider filters the returned detections in the browser.

## GitHub Pages

The static frontend is in `site/`. The workflow in `.github/workflows/pages.yml` publishes that directory. Select **GitHub Actions** as the Pages source in repository settings. The notebook, frontend and backend code live in this same repository.

## Model

The notebook starts from COCO pretrained Faster R-CNN with a ResNet-50 FPN backbone and trains the final predictor for the 20 PASCAL VOC classes. It saves the best checkpoint by validation mAP and evaluates once on the VOC 2007 test split. Use images you own or can redistribute for any permanent public sample gallery.
