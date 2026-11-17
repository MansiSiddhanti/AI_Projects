# AI Projects

A collection of machine learning projects I've worked on for practice. Each folder is a self-contained project with its own dependencies.

| Project | What it does | Main tools |
|---|---|---|
| [Chat with Your Documents](chat-with-your-documents/) | Ask questions about PDFs and get answers with page citations, fully local | Ollama (Llama 3.2), ChromaDB, sentence-transformers, Streamlit, FastAPI |
| [Tomato Leaf Disease Classifier](plant-leaf-disease-classifier/) | Classifies tomato leaf photos into 4 disease classes with Grad-CAM heatmaps | PyTorch (ResNet18), Streamlit |
| [Customer Churn Prediction](customer-churn-prediction/) | Predicts which telecom customers are likely to leave | pandas, scikit-learn, Jupyter |

## Getting started

```bash
git clone git@github.com:MansiSiddhanti/AI_Projects.git
cd AI_Projects
```

Each project uses its own virtual environment. You need Python 3.10 or newer.

### Chat with Your Documents

Needs [Ollama](https://ollama.com) installed. The PDFs are not in the repo, so put your own into `documents/` before ingesting.

```bash
cd chat-with-your-documents
ollama pull llama3.2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p documents          # copy your PDFs in here
python -m rag ingest
streamlit run app.py        # web UI on http://localhost:8501
```

See the [project README](chat-with-your-documents/README.md) for the REST API, CLI, Docker setup and evaluation.

### Tomato Leaf Disease Classifier

The trained model is included, so the app works right after installing.

```bash
cd plant-leaf-disease-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python predict.py assets/samples/early_blight_1.jpg   # or predict from the command line
```

Retraining needs the PlantVillage dataset. The steps are in the [project README](plant-leaf-disease-classifier/README.md).

### Customer Churn Prediction

The dataset (`data/Telco-Customer-Churn.csv`) and a saved model are included.

```bash
cd customer-churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook telco-customer-churn-on-icp4d-master/notebooks/Telco-customer-churn-ICP4D.ipynb
```

When running the notebook, skip the first `!pip install` cells. They install old IBM Watson package versions that are not needed for the local model.

The project is based on IBM's [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) sample. The `flaskapp/` folder comes from that sample and needs an IBM Cloud Pak for Data deployment to run.

## Repository layout

```
AI_Projects/
├── chat-with-your-documents/       RAG app over PDFs
├── plant-leaf-disease-classifier/  image classification
└── customer-churn-prediction/      tabular classification
```
