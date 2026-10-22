# Latent Dirichlet Allocation (LDA) Topic Modeling Studio

### Department of Artificial Intelligence and Machine Learning
**Advanced Natural Language Processing (ANLP) — Lab Experiment #4**  
*Global Academy of Technology / Autonomous Institution Affiliated to VTU*

---

## 🎯 Lab Objective
To apply **Latent Dirichlet Allocation (LDA)** on a document collection and interpret the discovered topics using probabilistic and sampling-based methods.

### 📋 Expected Outcomes & Deliverables
- [x] **Preprocess Text Data**: Tokenization, punctuation/URL removal, stopword filtering, POS tagging (`spaCy`), lemmatization, and n-gram collocation detection.
- [x] **Build Document-Term Representation**: Gensim Dictionary (Bag-of-Words) and Scikit-learn Document-Term Matrix (DTM) with vocabulary filtering & sparsity calculations.
- [x] **Train LDA Model**: Variational Bayes probabilistic inference via **Gensim** and **Scikit-learn** with Dirichlet hyperparameters ($\alpha$ and $\beta$).
- [x] **Interpret Topic-Word Distributions**: Extract Dirichlet word distributions ($\phi_k$) and document-topic distributions ($\theta_d$).
- [x] **Deliverables Produced**:
  1. Preprocessed corpus viewer & statistics
  2. Trained LDA model output (Log-perplexity & Coherence scores $C_v$ and $U_{mass}$)
  3. Topic-word probability distribution lists
  4. Academic Topic Interpretation Report (Automated synthesis exportable to Markdown and JSON)
  5. Interactive visualizations: Word Clouds, Intertopic Distance Map (2D MDS), Document-Topic Heatmap, and Coherence vs $k$ curve.
  6. Real-time Inference Playground for testing unseen documents.

---

## 🏗️ Project Architecture

```
AEGIS-ALPHA/
├── app.py                         # Flask web application & REST API
├── run_lda_cli.py                 # Command-line interface for batch execution
├── test_lda.py                    # Automated unit test suite
├── ANLP_LDA_Lab_Report.ipynb      # Complete Jupyter Notebook for lab submissions
├── requirements.txt               # Dependencies
├── lda_engine/
│   ├── __init__.py
│   ├── preprocessor.py            # Text cleaning, spaCy/NLTK lemmatization, n-grams
│   ├── vectorizer.py              # Document-Term representation & matrix sparsity
│   ├── lda_model.py               # Gensim & Scikit-learn LDA wrappers
│   ├── evaluation.py              # Coherence (Cv, Umass), perplexity, optimal k search
│   ├── visualizer.py              # Word clouds (base64), MDS intertopic distance, pyLDAvis
│   └── interpreter.py             # Topic labeling, interpretation report generator
├── data/
│   ├── sample_tech_science_news.json # 16 curated documents (AI, Space, Medicine, Energy)
│   └── sample_customer_reviews.json  # 12 customer feedback reviews
├── static/
│   ├── css/style.css              # Custom styling, dark mode, typography
│   └── js/main.js                 # Interactive dashboard controller & Plotly charts
└── templates/
    └── index.html                 # Clean, modern single-page dashboard UI
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Click **"1-Click Auto Run"** for instant end-to-end execution.
- Toggle between tabs to inspect the preprocessed corpus, DTM statistics, hyperparameter tuning, visualizations, topic reports, and the live inference tester.

### 3. Run Headless via CLI
You can run topic modeling directly in the terminal:
```bash
# Basic run with 4 topics
python run_lda_cli.py --topics 4

# Run with automated grid search for optimal k (C_v coherence optimization)
python run_lda_cli.py --find-optimal-k --output-report Lab_Interpretation_Report.md
```

### 4. Run the Jupyter Notebook
For lab report write-ups and viva demonstration:
```bash
jupyter lab ANLP_LDA_Lab_Report.ipynb
# or
jupyter notebook ANLP_LDA_Lab_Report.ipynb
```

### 5. Run the Automated Tests
```bash
python -m unittest test_lda.py -v
```

---

## 🔬 Mathematical Formulation
- **Document-Topic Distribution:** $\boldsymbol{\theta}_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$
- **Topic-Word Distribution:** $\boldsymbol{\phi}_k \sim \text{Dirichlet}(\boldsymbol{\beta})$
- **Generative Process:** For each document $d$, sample topic proportions $\theta_d$; for each word position $n$, sample topic assignment $z_{d,n} \sim \text{Multinomial}(\theta_d)$ and term $w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}})$.
- **Inference:** Variational Bayes / Collapsed Gibbs Sampling approximates the posterior $P(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\phi} \mid \mathbf{w}, \alpha, \beta)$.
