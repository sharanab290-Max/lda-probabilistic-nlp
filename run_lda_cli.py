"""
Command-Line Interface (CLI) for LDA Topic Modeling
Executes corpus preprocessing, document-term matrix construction,
LDA model fitting, coherence scoring, and topic report generation.
"""

import os
import sys
import argparse
import json
from lda_engine.preprocessor import TextPreprocessor
from lda_engine.vectorizer import DocumentTermMatrix
from lda_engine.lda_model import LDATopicModel
from lda_engine.evaluation import evaluate_topic_coherence, find_optimal_topics
from lda_engine.interpreter import TopicInterpreter


def main():
    parser = argparse.ArgumentParser(description="Latent Dirichlet Allocation Topic Modeling CLI")
    parser.add_argument("--corpus", type=str, default="data/sample_tech_science_news.json", help="Path to JSON or text corpus")
    parser.add_argument("--topics", type=int, default=4, help="Number of latent topics k")
    parser.add_argument("--engine", type=str, choices=["gensim", "sklearn"], default="gensim", help="LDA inference engine")
    parser.add_argument("--passes", type=int, default=15, help="Number of training passes/epochs")
    parser.add_argument("--find-optimal-k", action="store_true", help="Run grid search for optimal k topic count")
    parser.add_argument("--output-report", type=str, default="LDA_Interpretation_Report.md", help="Output Markdown report path")
    args = parser.parse_args()

    print("=" * 70)
    print("  LATENT DIRICHLET ALLOCATION (LDA) TOPIC MODELING SYSTEM")
    print("  Global Academy of Technology - Dept of AI & ML (ANLP Lab)")
    print("=" * 70)

    # 1. Ingest Corpus
    if not os.path.exists(args.corpus):
        print(f"[-] Error: Corpus file '{args.corpus}' not found.")
        sys.exit(1)

    print(f"\n[+] Loading corpus from: {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs = [item.get("text", "") for item in data] if isinstance(data, list) else []

    print(f"    Loaded {len(docs)} documents.")

    # 2. Preprocess Text Data
    print("\n[+] Step 1: Preprocessing Corpus (Tokenization, POS filtering, Lemmatization)...")
    preprocessor = TextPreprocessor(use_spacy=True)
    prep_result = preprocessor.preprocess_corpus(docs, build_bigrams=True)
    tokens = prep_result["tokens"]
    clean_texts = prep_result["clean_texts"]
    stats = prep_result["stats"]

    print(f"    Total Tokens: {stats['total_tokens']}")
    print(f"    Vocabulary Size: {stats['vocabulary_size']}")
    print(f"    Bigrams Discovered: {stats['bigrams_detected']}")
    print(f"    Avg Doc Length: {stats['avg_tokens_per_doc']} tokens")

    # 3. Document-Term Matrix
    print("\n[+] Step 2: Constructing Document-Term Representation (Bag-of-Words & DTM)...")
    dtm = DocumentTermMatrix(no_below=1, no_above=0.95)
    dtm_stats = dtm.fit_transform(tokens, clean_texts)
    print(f"    Matrix Sparsity: {dtm_stats['matrix_sparsity_percent']}%")
    print(f"    Top Terms: {', '.join([t['word'] for t in dtm_stats['top_terms'][:6]])}")

    # 4. Optional Grid Search for Optimal k
    if args.find_optimal_k and args.engine == "gensim":
        print("\n[+] Evaluating Topic Coherence (C_v) across k in [2, 6]...")
        opt_results = find_optimal_topics(tokens, dtm, min_k=2, max_k=6)
        print(f"    Optimal k: {opt_results['optimal_k']} (C_v = {opt_results['best_coherence']})")
        for k, c, p in zip(opt_results['k_values'], opt_results['coherence_scores'], opt_results['perplexity_scores']):
            print(f"      k={k} -> Coherence C_v: {c:.4f} | Perplexity: {p:.2f}")
        args.topics = opt_results['optimal_k']

    # 5. Train LDA Model
    print(f"\n[+] Step 3: Training {args.engine.upper()} LDA Model with k={args.topics} topics...")
    model = LDATopicModel(
        num_topics=args.topics,
        engine=args.engine,
        passes=args.passes,
        iterations=100
    )
    summary = model.fit(dtm)

    coherence = {}
    if args.engine == "gensim" and model.gensim_model:
        coherence = evaluate_topic_coherence(
            model=model.gensim_model,
            token_lists=tokens,
            dictionary=dtm.dictionary
        )
        print(f"    Topic Coherence (C_v): {coherence.get('c_v', 'N/A')}")
        print(f"    Topic Coherence (U_mass): {coherence.get('u_mass', 'N/A')}")

    print(f"    Model Perplexity: {summary.get('perplexity', 'N/A')}")

    # 6. Display Topic-Word Distributions
    print("\n[+] Step 4: Discovered Topic-Word Distributions (Beta):")
    for t in model.topic_words:
        top_words = [f"{w['word']}({w['weight']:.3f})" for w in t['top_words'][:6]]
        print(f"    Topic #{t['topic_id'] + 1}: {', '.join(top_words)}")

    # 7. Generate Topic Interpretation Report
    print(f"\n[+] Step 5: Generating Academic Interpretation Report -> {args.output_report}")
    report = TopicInterpreter.generate_interpretation_report(
        topic_words=model.topic_words,
        doc_topics=model.doc_topics,
        corpus_stats=dtm_stats,
        model_stats=summary,
        coherence_stats=coherence,
        sample_documents=docs
    )

    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write(report["markdown_report"])

    print(f"    Report saved successfully to {os.path.abspath(args.output_report)}")
    print("\n[SUCCESS] All ANLP Lab Deliverables successfully produced!")


if __name__ == "__main__":
    main()
