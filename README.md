# Neuro-Somatic AI

Streamlit application for classifying medical images with an Ultralytics YOLO classification model.

## Project layout

```text
models/best.pt       Trained model
train/               Training dataset (ignored by Git)
validation/          Validation dataset (ignored by Git)
test/                Test dataset (ignored by Git)
test_inference.py    Command-line inference script
app.py               Streamlit web application
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Ensure the trained model is available at `models/best.pt`.

## Run local inference

Run against the first compatible image found under `test/`:

```bash
python test_inference.py
```

Or provide a specific image path:

```bash
python test_inference.py test/Normal_MRI/example.jpg
```

## Run the Streamlit dashboard

```bash
streamlit run app.py
```