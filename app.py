"""
Latent Dirichlet Allocation (LDA) Interactive Web Application
Department of Artificial Intelligence and Machine Learning
Global Academy of Technology
"""

import os
import json
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, send_file, Response
from lda_engine.preprocessor import TextPreprocessor
from lda_engine.vectorizer import DocumentTermMatrix
from lda_engine.lda_model import LDATopicModel
from lda_engine.evaluation import evaluate_topic_coherence, find_optimal_topics
from lda_engine.interpreter import TopicInterpreter
from lda_engine.visualizer import LDAVisualizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'anlp-lda-secret-key-2026'

# In-memory session state for active corpus & trained model
APP_STATE = {
    "raw_documents": [],
    "document_titles": [],
    "clean_corpus": None,
    "tokens": None,
    "dtm": None,
    "dtm_stats": None,
    "lda_model": None,
    "model_stats": None,
    "coherence_stats": None,
    "topic_words": None,
    "doc_topics": None,
    "interpretation_report": None,
    "wordclouds": [],
    "distance_map": [],
    "active_dataset_name": "Tech & Science News"
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_default_dataset(name: str = "sample_tech_science_news.json"):
    filepath = os.path.join(DATA_DIR, name)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            docs = [item.get("text", "") for item in data]
            titles = [item.get("title", f"Doc #{i+1}") for i, item in enumerate(data)]
            return docs, titles
    return [], []


# Preload default dataset on startup
default_docs, default_titles = load_default_dataset()
APP_STATE["raw_documents"] = default_docs
APP_STATE["document_titles"] = default_titles


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = [
        {
            "id": "sample_tech_science_news.json",
            "name": "Science, AI, Medicine & Energy",
            "description": "16 curated articles across AI, Space, Oncology, and Green Energy.",
            "doc_count": 16
        },
        {
            "id": "sample_customer_reviews.json",
            "name": "E-Commerce Customer Reviews",
            "description": "12 customer feedback reviews covering support, shipping, quality, and pricing.",
            "doc_count": 12
        }
    ]
    return jsonify({
        "status": "success",
        "datasets": datasets,
        "active_dataset": APP_STATE["active_dataset_name"],
        "current_doc_count": len(APP_STATE["raw_documents"])
    })


@app.route('/api/load-dataset', methods=['POST'])
def load_dataset():
    data = request.get_json() or {}
    dataset_id = data.get("dataset_id", "sample_tech_science_news.json")

    docs, titles = load_default_dataset(dataset_id)
    if not docs:
        return jsonify({"status": "error", "message": "Dataset not found"}), 404

    APP_STATE["raw_documents"] = docs
    APP_STATE["document_titles"] = titles
    APP_STATE["active_dataset_name"] = "Tech & Science News" if "tech" in dataset_id else "Customer Reviews"
    # Reset downstream steps
    APP_STATE["tokens"] = None
    APP_STATE["dtm"] = None
    APP_STATE["lda_model"] = None

    return jsonify({
        "status": "success",
        "message": f"Successfully loaded {len(docs)} documents.",
        "documents": [
            {"id": i, "title": titles[i], "preview": docs[i][:150] + "..."}
            for i in range(len(docs))
        ]
    })


@app.route('/api/upload-corpus', methods=['POST'])
def upload_corpus():
    try:
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename.lower()
            content = file.read().decode('utf-8', errors='ignore')

            docs = []
            titles = []

            if filename.endswith('.json'):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for i, item in enumerate(parsed):
                        if isinstance(item, dict):
                            docs.append(item.get("text", str(item)))
                            titles.append(item.get("title", f"Doc #{i+1}"))
                        else:
                            docs.append(str(item))
                            titles.append(f"Doc #{i+1}")
            elif filename.endswith('.csv'):
                import csv
                import io
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                if rows:
                    header = rows[0]
                    text_idx = 0
                    for idx, col in enumerate(header):
                        if col.lower() in ['text', 'content', 'document', 'body', 'review']:
                            text_idx = idx
                            break
                    for r_idx, row in enumerate(rows[1:]):
                        if len(row) > text_idx and row[text_idx].strip():
                            docs.append(row[text_idx].strip())
                            titles.append(f"Row #{r_idx+1}")
            else:
                # Plain text: split by blank lines or newlines
                raw_paras = [p.strip() for p in content.split('\n\n') if p.strip()]
                if len(raw_paras) > 1:
                    docs = raw_paras
                else:
                    docs = [line.strip() for line in content.splitlines() if line.strip()]
                titles = [f"Text Segment #{i+1}" for i in range(len(docs))]

            if not docs:
                return jsonify({"status": "error", "message": "No valid text content found in file."}), 400

            APP_STATE["raw_documents"] = docs
            APP_STATE["document_titles"] = titles
            APP_STATE["active_dataset_name"] = file.filename
            APP_STATE["tokens"] = None
            APP_STATE["dtm"] = None
            APP_STATE["lda_model"] = None

            return jsonify({
                "status": "success",
                "message": f"Successfully parsed {len(docs)} documents from {file.filename}.",
                "documents": [
                    {"id": i, "title": titles[i], "preview": docs[i][:150] + "..."}
                    for i in range(len(docs))
                ]
            })

        # Plain text pasted directly in JSON payload
        data = request.get_json() or {}
        custom_text = data.get("text", "")
        if custom_text:
            raw_paras = [p.strip() for p in custom_text.split('\n\n') if p.strip()]
            if len(raw_paras) <= 1:
                raw_paras = [l.strip() for l in custom_text.splitlines() if l.strip()]

            if not raw_paras:
                return jsonify({"status": "error", "message": "Text is empty."}), 400

            APP_STATE["raw_documents"] = raw_paras
            APP_STATE["document_titles"] = [f"Custom Doc #{i+1}" for i in range(len(raw_paras))]
            APP_STATE["active_dataset_name"] = "Custom User Input"
            APP_STATE["tokens"] = None
            APP_STATE["dtm"] = None
            APP_STATE["lda_model"] = None

            return jsonify({
                "status": "success",
                "message": f"Loaded {len(raw_paras)} documents from custom input.",
                "documents": [
                    {"id": i, "title": APP_STATE["document_titles"][i], "preview": raw_paras[i][:150] + "..."}
                    for i in range(len(raw_paras))
                ]
            })

        return jsonify({"status": "error", "message": "No file or text provided."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json() or {}
    use_spacy = bool(data.get("use_spacy", True))
    pos_filter = data.get("pos_filter", ["NOUN", "ADJ", "VERB", "ADV"])
    min_word_length = int(data.get("min_word_length", 3))
    custom_stopwords = data.get("custom_stopwords", [])
    build_bigrams = bool(data.get("build_bigrams", True))

    if not APP_STATE["raw_documents"]:
        return jsonify({"status": "error", "message": "No documents loaded to preprocess."}), 400

    preprocessor = TextPreprocessor(
        use_spacy=use_spacy,
        pos_filter=pos_filter,
        min_word_length=min_word_length,
        custom_stopwords=custom_stopwords
    )

    result = preprocessor.preprocess_corpus(
        documents=APP_STATE["raw_documents"],
        build_bigrams=build_bigrams
    )

    APP_STATE["tokens"] = result["tokens"]
    APP_STATE["clean_corpus"] = result["clean_texts"]

    # Build Document-Term representation
    dtm = DocumentTermMatrix(no_below=1, no_above=0.95)
    dtm_stats = dtm.fit_transform(result["tokens"], result["clean_texts"])

    APP_STATE["dtm"] = dtm
    APP_STATE["dtm_stats"] = dtm_stats

    # Sample preprocessed previews
    previews = []
    for i in range(min(5, len(result["tokens"]))):
        previews.append({
            "doc_id": i,
            "title": APP_STATE["document_titles"][i] if i < len(APP_STATE["document_titles"]) else f"Doc #{i+1}",
            "original_length": len(APP_STATE["raw_documents"][i].split()),
            "token_count": len(result["tokens"][i]),
            "tokens_sample": result["tokens"][i][:12]
        })

    return jsonify({
        "status": "success",
        "stats": {
            **result["stats"],
            **dtm_stats
        },
        "previews": previews
    })


@app.route('/api/train-lda', methods=['POST'])
def train_lda():
    if not APP_STATE["dtm"] or not APP_STATE["tokens"]:
        return jsonify({"status": "error", "message": "Please preprocess text data first."}), 400

    data = request.get_json() or {}
    num_topics = int(data.get("num_topics", 4))
    engine = data.get("engine", "gensim")
    alpha = data.get("alpha", "symmetric")
    beta = data.get("beta", None)
    passes = int(data.get("passes", 15))
    iterations = int(data.get("iterations", 100))

    model = LDATopicModel(
        num_topics=num_topics,
        engine=engine,
        alpha=alpha,
        beta=beta,
        passes=passes,
        iterations=iterations
    )

    model_summary = model.fit(APP_STATE["dtm"])
    APP_STATE["lda_model"] = model
    APP_STATE["model_stats"] = model_summary
    APP_STATE["topic_words"] = model.topic_words
    APP_STATE["doc_topics"] = model.doc_topics

    # Calculate Coherence if Gensim engine used
    coherence_stats = {}
    if engine == "gensim" and model.gensim_model:
        coherence_stats = evaluate_topic_coherence(
            model=model.gensim_model,
            token_lists=APP_STATE["tokens"],
            dictionary=APP_STATE["dtm"].dictionary,
            coherence_types=['c_v', 'u_mass']
        )
    APP_STATE["coherence_stats"] = coherence_stats

    # Generate Visuals
    wordclouds = LDAVisualizer.generate_wordclouds_base64(model.topic_words)
    APP_STATE["wordclouds"] = wordclouds

    distance_map = LDAVisualizer.compute_intertopic_distance_map(
        topic_words=model.topic_words,
        doc_topics=model.doc_topics,
        dtm_matrix=APP_STATE["dtm"].dtm_matrix
    )
    APP_STATE["distance_map"] = distance_map

    # Generate Topic Interpretation Report
    report = TopicInterpreter.generate_interpretation_report(
        topic_words=model.topic_words,
        doc_topics=model.doc_topics,
        corpus_stats=APP_STATE["dtm_stats"],
        model_stats=model_summary,
        coherence_stats=coherence_stats,
        sample_documents=APP_STATE["raw_documents"]
    )
    APP_STATE["interpretation_report"] = report

    # Update topic labels with inferred semantic titles
    for i, t in enumerate(model.topic_words):
        t["label"] = report["topic_analyses"][i]["label"]

    return jsonify({
        "status": "success",
        "model_summary": model_summary,
        "coherence_stats": coherence_stats,
        "topic_words": model.topic_words,
        "doc_topics": model.doc_topics,
        "wordclouds": wordclouds,
        "distance_map": distance_map,
        "report": report
    })


@app.route('/api/evaluate-k', methods=['POST'])
def evaluate_k():
    if not APP_STATE["dtm"] or not APP_STATE["tokens"]:
        return jsonify({"status": "error", "message": "Corpus must be preprocessed first."}), 400

    data = request.get_json() or {}
    min_k = int(data.get("min_k", 2))
    max_k = int(data.get("max_k", 6))

    eval_results = find_optimal_topics(
        token_lists=APP_STATE["tokens"],
        dtm=APP_STATE["dtm"],
        min_k=min_k,
        max_k=max_k,
        step=1
    )

    return jsonify({
        "status": "success",
        "evaluation": eval_results
    })


@app.route('/api/infer-text', methods=['POST'])
def infer_text():
    if not APP_STATE["lda_model"]:
        return jsonify({"status": "error", "message": "No trained LDA model available. Train a model first."}), 400

    data = request.get_json() or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"status": "error", "message": "Input text cannot be empty."}), 400

    preprocessor = TextPreprocessor(use_spacy=False)
    tokens = preprocessor.preprocess_document(text)

    if not tokens:
        return jsonify({"status": "error", "message": "No recognizable vocabulary words in the input text."}), 400

    prediction = APP_STATE["lda_model"].infer_document(tokens)

    # Attach inferred human-readable label if report is available
    if APP_STATE["interpretation_report"]:
        dom_idx = prediction["dominant_topic"]
        if dom_idx < len(APP_STATE["interpretation_report"]["topic_analyses"]):
            prediction["dominant_label"] = APP_STATE["interpretation_report"]["topic_analyses"][dom_idx]["label"]
        for item in prediction["distribution"]:
            t_id = item["topic_id"]
            if t_id < len(APP_STATE["interpretation_report"]["topic_analyses"]):
                item["label"] = APP_STATE["interpretation_report"]["topic_analyses"][t_id]["label"]

    return jsonify({
        "status": "success",
        "tokens": tokens,
        "prediction": prediction
    })


@app.route('/api/export-report', methods=['GET'])
def export_report():
    if not APP_STATE["interpretation_report"]:
        return jsonify({"status": "error", "message": "No interpretation report generated yet."}), 400

    export_format = request.args.get("format", "markdown")
    if export_format == "json":
        return Response(
            json.dumps(APP_STATE["interpretation_report"], indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=LDA_Topic_Report.json"}
        )
    else:
        return Response(
            APP_STATE["interpretation_report"]["markdown_report"],
            mimetype="text/markdown",
            headers={"Content-Disposition": "attachment;filename=LDA_Topic_Report.md"}
        )


if __name__ == '__main__':
    print("Starting Latent Dirichlet Allocation (LDA) Topic Modeling System...")
    app.run(host='127.0.0.1', port=5000, debug=True)
