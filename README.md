# Inside the Detector

An interactive Faster R-CNN exhibit and the executed notebook that trains its PASCAL VOC 2007 model.

The attached full run selected epoch 10 with 81.7% validation mAP and reported 81.8% test mAP on 4,952 VOC 2007 test images. Its warm median inference time was 24 ms on an NVIDIA A100-SXM4-40GB. These values come from the notebook output and have not been measured on the eventual demo host.

## Run the website

Serve the repository root, then open `/site/`. For example, `python -m http.server 8000` and visit `http://localhost:8000/site/`.

The two bundled VOC scenes show **real saved detections** from the trained checkpoint. Uploads run the exported model on your own device through ONNX Runtime Web in a worker. The first upload downloads roughly 166 MB of model weights. Later uploads reuse the loaded model until the page closes. Desktop browsers are recommended. The animated stages explain inference and do not display live internal activations.

## Train and export the detector

1. Run `notebooks/FasterRCNN_VOC2007_Portfolio.ipynb` in Colab with a GPU and `MODE = "full"`.
2. Download the exported `voc-detector-full.zip` and extract `best.pt`.
3. Install compatible PyTorch and TorchVision packages, plus `onnx`, Pillow, and NumPy.
4. Run `python scripts/export_browser.py /path/to/best.pt`. It exports a fixed 640 × 640 ONNX graph and splits its weights into three files below GitHub's regular file limit.
5. Push the updated website and model files to publish.

The browser preserves aspect ratio with letterbox padding, then maps boxes back to original image coordinates. The fixed browser preprocessing differs from the original notebook evaluation, so browser predictions can differ. The exported graph passed numerical parity checks against PyTorch on the same fixed input. The original PyTorch inference module and optional FastAPI service remain in `backend/`.

## GitHub Pages

The static frontend is in `site/`. The workflow in `.github/workflows/pages.yml` publishes that directory. Select **GitHub Actions** as the Pages source in repository settings. The notebook, frontend and backend code live in this same repository.

## Model

The notebook starts from COCO pretrained Faster R-CNN with a ResNet-50 FPN backbone and trains the final predictor for the 20 PASCAL VOC classes. It saves the best checkpoint by validation mAP and evaluates once on the VOC 2007 test split. Use images you own or can redistribute for any permanent public sample gallery.

## Author

- GitHub: [@AbdulrahmanAKh](https://github.com/AbdulrahmanAKh)
- LinkedIn: [Abdulrahman Alkhatib](https://www.linkedin.com/in/abdulrahman-alkhatib/)
- Email: [aboodkhatib90@gmail.com](mailto:aboodkhatib90@gmail.com)
- Portfolio: [abdulrahmanakh.github.io](https://abdulrahmanakh.github.io/)
- Notebook URL: [YOLO SiteSight Project](https://AbdulrahmanAKh.github.io/YOLO_SiteSight_Project)

**Developed & Engineered by Abdulrahman Alkhatib**  
*Computer Science Graduate | Software & AI Engineer*
