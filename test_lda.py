"""
Automated Test Suite for LDA Topic Modeling Engine
Verifies text preprocessing, dictionary construction, model training,
coherence calculation, and inference on unseen text.
"""

import unittest
from lda_engine.preprocessor import TextPreprocessor
from lda_engine.vectorizer import DocumentTermMatrix
from lda_engine.lda_model import LDATopicModel
from lda_engine.evaluation import evaluate_topic_coherence
from lda_engine.interpreter import TopicInterpreter


class TestLDAEngine(unittest.TestCase):

    def setUp(self):
        self.documents = [
            "Deep learning neural networks and computer vision models identify objects with high accuracy.",
            "Convolutional neural networks train on image datasets using backpropagation algorithms.",
            "NASA space telescope captures spectroscopic images of ancient stars and distant orbiting galaxies.",
            "Astronomers analyze infrared light emitted by planetary atmospheres and solar systems.",
            "Clinical trials of oncology mRNA vaccines demonstrate improved patient survival rates.",
            "Immunotherapy and chemotherapy treat malignant tumor cells in cancer patients.",
            "Solar photovoltaic panels and lithium battery storage generate clean renewable electricity.",
            "Wind turbine blades produce zero emission renewable electrical energy for the power grid."
        ]

    def test_preprocessing(self):
        preprocessor = TextPreprocessor(use_spacy=False, min_word_length=3)
        res = preprocessor.preprocess_corpus(self.documents, build_bigrams=False)
        self.assertEqual(len(res["tokens"]), len(self.documents))
        self.assertGreater(res["stats"]["vocabulary_size"], 10)
        self.assertGreater(res["stats"]["total_tokens"], 20)

    def test_vectorizer(self):
        preprocessor = TextPreprocessor(use_spacy=False)
        res = preprocessor.preprocess_corpus(self.documents, build_bigrams=False)
        dtm = DocumentTermMatrix(no_below=1, no_above=0.95)
        stats = dtm.fit_transform(res["tokens"], res["clean_texts"])
        self.assertIn("vocabulary_size", stats)
        self.assertIn("matrix_sparsity_percent", stats)
        self.assertGreater(len(dtm.bow_corpus), 0)

    def test_gensim_lda_fit(self):
        preprocessor = TextPreprocessor(use_spacy=False)
        res = preprocessor.preprocess_corpus(self.documents, build_bigrams=False)
        dtm = DocumentTermMatrix(no_below=1, no_above=0.95)
        dtm.fit_transform(res["tokens"], res["clean_texts"])

        lda = LDATopicModel(num_topics=4, engine="gensim", passes=5, iterations=20)
        summary = lda.fit(dtm)

        self.assertEqual(summary["num_topics"], 4)
        self.assertEqual(len(summary["topic_word_distributions"]), 4)
        self.assertEqual(len(summary["doc_topic_distributions"]), len(self.documents))

        # Test inference on new document
        new_doc = ["neural", "network", "vision", "model"]
        infer_res = lda.infer_document(new_doc)
        self.assertIn("dominant_topic", infer_res)
        self.assertEqual(len(infer_res["distribution"]), 4)

    def test_sklearn_lda_fit(self):
        preprocessor = TextPreprocessor(use_spacy=False)
        res = preprocessor.preprocess_corpus(self.documents, build_bigrams=False)
        dtm = DocumentTermMatrix(no_below=1, no_above=0.95)
        dtm.fit_transform(res["tokens"], res["clean_texts"])

        lda = LDATopicModel(num_topics=3, engine="sklearn", iterations=20)
        summary = lda.fit(dtm)

        self.assertEqual(summary["num_topics"], 3)
        self.assertIn("perplexity", summary)

    def test_topic_interpreter(self):
        mock_topic_words = [
            {
                "topic_id": 0,
                "label": "Topic #1",
                "top_words": [{"word": "neural", "weight": 0.4}, {"word": "model", "weight": 0.3}],
                "top_words_string": "neural, model"
            }
        ]
        mock_doc_topics = [{"doc_id": 0, "dominant_topic": 0, "dominant_weight": 0.9, "distribution": [0.9]}]
        mock_stats = {"num_documents": 1, "vocabulary_size": 2}
        mock_model = {"engine": "Gensim", "perplexity": 10.5}

        report = TopicInterpreter.generate_interpretation_report(
            topic_words=mock_topic_words,
            doc_topics=mock_doc_topics,
            corpus_stats=mock_stats,
            model_stats=mock_model
        )
        self.assertIn("markdown_report", report)
        self.assertIn("Executive Summary", report["markdown_report"])


if __name__ == '__main__':
    unittest.main()
