import sys
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLabel, QPushButton,
                             QComboBox, QSplitter, QFrame, QProgressBar,
                             QFileDialog, QTabWidget, QScrollArea, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from transformers import pipeline
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from statement_similarity_analysis import (analyze_statements, extract_key_points,
                                           compare_key_points, calculate_confidence_score,
                                           generate_detailed_explanation, create_visualizations)
from statement_similarity_analysis import compare_key_points, extract_key_points  # ✅ ekstra import


class ModelLoader:
    _instance = None
    _similarity_model = None
    _cross_model = None
    _nlp = None

    @classmethod
    def get_similarity_model(cls):
        if cls._similarity_model is None:
            cls._similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return cls._similarity_model

    @classmethod
    def get_cross_model(cls):
        if cls._cross_model is None:
            cls._cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return cls._cross_model

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load("en_core_web_sm")
        return cls._nlp


def semantic_consistency_score(sentence1, sentence2):
    try:
        cross_model = ModelLoader.get_cross_model()
        score = cross_model.predict([(sentence1, sentence2)])[0]
        return float(score)
    except Exception as e:
        print(f"CrossEncoder error: {e}")
        return 0.0


class KeywordHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keywords = []
        self.format = QTextCharFormat()
        self.format.setBackground(QColor("#FFEB3B"))

    def set_keywords(self, keywords):
        self.keywords = keywords
        self.rehighlight()

    def highlightBlock(self, text):
        for keyword in self.keywords:
            index = 0
            while index >= 0:
                index = text.find(keyword, index)
                if index >= 0:
                    self.setFormat(index, len(keyword), self.format)
                    index += len(keyword)


class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, case, similarity_model, nlp):
        super().__init__()
        self.case = case
        self.similarity_model = similarity_model
        self.nlp = nlp

    def run(self):
        results = {}
        total_steps = len(self.case['witness_statements']) + 1
        current_step = 0

        witness_statements = self.case['witness_statements']
        suspect_statement = self.case['suspect_statement']

        for i, witness in enumerate(witness_statements):
            analysis = analyze_statements(suspect_statement, witness)
            deep_semantic_score = semantic_consistency_score(suspect_statement, witness)
            combined_score = (0.6 * analysis['similarity']) + (0.4 * deep_semantic_score)

            results[f'witness_{i + 1}'] = {
                'statement': witness,
                'similarity': combined_score,
                'confidence': analysis['confidence'],
                'comparison': analysis['comparison'],
                'cross_score': deep_semantic_score,
                'weighted_score': analysis['similarity'],
                'explanation': generate_detailed_explanation(analysis, suspect_statement, witness),
                'result': "✅ Consistent/Supportive" if combined_score > 0.6 and analysis['confidence'] > 0.5 else
                "❌ Contradictory" if combined_score < 0.3 or analysis['confidence'] < 0.3 else
                "⚠️ Unclear/Neutral"
            }

            current_step += 1
            self.progress.emit(int(current_step * 100 / total_steps))

        all_text = suspect_statement + " " + " ".join(witness_statements)
        doc = self.nlp(all_text)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]
        results['keywords'] = [word for word, count in Counter(keywords).most_common(10)]

        create_visualizations([results[f'witness_{i + 1}'] for i in range(len(witness_statements))],
                              self.case['description'])

        current_step += 1
        self.progress.emit(int(current_step * 100 / total_steps))

        self.finished.emit(results)


class InterrogationAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interrogation Statement Analyzer")
        self.setGeometry(100, 100, 1400, 900)

        with open("forensic_statements_data_en.json", encoding="utf-8") as f:
            self.data = json.load(f)

        self.init_ui()
        self.update_analysis(0)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        controls_layout = QHBoxLayout()
        self.case_selector = QComboBox()
        self.case_selector.addItems([case['description'] for case in self.data])
        self.case_selector.currentIndexChanged.connect(self.update_analysis)
        controls_layout.addWidget(QLabel("Select Case:"))
        controls_layout.addWidget(self.case_selector)

        self.export_btn = QPushButton("Export Analysis")
        self.export_btn.clicked.connect(self.export_analysis)
        controls_layout.addWidget(self.export_btn)

        layout.addLayout(controls_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Suspect Statement:"))
        self.suspect_text = QTextEdit()
        self.suspect_text.setReadOnly(True)
        left_layout.addWidget(self.suspect_text)
        left_layout.addWidget(QLabel("Summary:"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        left_layout.addWidget(self.summary_text)
        self.summary_btn = QPushButton("Generate Summary")
        self.summary_btn.clicked.connect(self.generate_summary)
        left_layout.addWidget(self.summary_btn)

        content_splitter.addWidget(left_panel)

        right_panel = QTabWidget()

        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.addWidget(QLabel("Witness Statements:"))
        self.witness_text = QTextEdit()
        self.witness_text.setReadOnly(True)
        analysis_layout.addWidget(self.witness_text)
        analysis_layout.addWidget(QLabel("Analysis Results:"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        right_panel.addTab(analysis_widget, "Analysis")

        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        timeline_layout.addWidget(self.canvas)
        right_panel.addTab(timeline_widget, "Timeline")

        keywords_widget = QWidget()
        keywords_layout = QVBoxLayout(keywords_widget)
        self.keywords_text = QTextEdit()
        self.keywords_text.setReadOnly(True)
        keywords_layout.addWidget(self.keywords_text)
        right_panel.addTab(keywords_widget, "Keywords")

        # --- NEW DIFFERENCES TAB ---
        differences_widget = QWidget()
        differences_layout = QVBoxLayout(differences_widget)
        self.differences_text = QTextEdit()
        self.differences_text.setReadOnly(True)
        differences_layout.addWidget(self.differences_text)
        right_panel.addTab(differences_widget, "Differences")

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([400, 1000])
        layout.addWidget(content_splitter)

    def update_analysis(self, index):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        case = self.data[index]
        self.suspect_text.setText(case['suspect_statement'])

        self.worker = AnalysisWorker(case, ModelLoader.get_similarity_model(), ModelLoader.get_nlp())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.update_ui_with_results)
        self.worker.start()

    def update_ui_with_results(self, results):
        witness_text = ""
        analysis_text = ""
        difference_text = ""

        case = self.data[self.case_selector.currentIndex()]
        suspect_statement = case["suspect_statement"]

        for i in range(1, len(results)):
            if f'witness_{i}' in results:
                w_data = results[f'witness_{i}']
                witness_text += f"Witness {i}:\n{w_data['statement']}\n\n"
                analysis_text += f"Witness {i} Analysis:\n"
                analysis_text += f"Weighted Similarity: {w_data['weighted_score']:.2f}\n"
                analysis_text += f"Deep Semantic Score: {w_data['cross_score']:.2f}\n"
                analysis_text += f"Final Similarity: {w_data['similarity']:.2f}\n"
                analysis_text += f"Confidence: {w_data['confidence']:.2f}\n"
                analysis_text += f"Result: {w_data['result']}\n\n"
                analysis_text += "Detailed Explanation:\n" + w_data['explanation'] + "\n\n"

                witness_statement = w_data["statement"]
                s_points = extract_key_points(suspect_statement)
                w_points = extract_key_points(witness_statement)
                comp = compare_key_points(s_points, w_points)
                difference_text += f"🧾 Witness {i} vs Suspect\n" + "-" * 60 + "\n"
                for cat, vals in comp.items():
                    match, diff = vals["matching"], vals["different"]
                    if match or diff:
                        difference_text += f"📘 {cat.upper()}:\n"
                        if match:
                            difference_text += f"   ✅ Matching: {', '.join(match)}\n"
                        if diff:
                            difference_text += f"   ❌ Different: {', '.join(diff)}\n"
                difference_text += "\n"

        self.witness_text.setText(witness_text)
        self.analysis_text.setText(analysis_text)
        self.differences_text.setText(difference_text)

        if 'keywords' in results:
            self.keywords_text.setText("\n".join(results['keywords']))

        self.update_timeline(results)
        self.progress_bar.setVisible(False)

    def update_timeline(self, results):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        statements = ['Suspect'] + [f'Witness {i + 1}' for i in range(len(results) - 2)]
        similarities = [1.0] + [results[f'witness_{i + 1}']['similarity'] for i in range(len(results) - 2)]
        ax.plot(range(len(statements)), similarities, 'bo-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(statements)))
        ax.set_xticklabels(statements, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Statement Similarity Timeline')
        self.figure.tight_layout()
        self.canvas.draw()

    def export_analysis(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Analysis", "", "CSV Files (*.csv);;JSON Files (*.json)")
        if file_name:
            case = self.data[self.case_selector.currentIndex()]
            analysis_data = {
                'case_description': case['description'],
                'suspect_statement': case['suspect_statement'],
                'witness_statements': case['witness_statements'],
                'analysis': self.analysis_text.toPlainText(),
                'differences': self.differences_text.toPlainText(),
                'keywords': self.keywords_text.toPlainText(),
                'summary': getattr(self, "generated_summary", "No summary generated"),
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                if file_name.endswith('.csv'):
                    pd.DataFrame([analysis_data]).to_csv(file_name, index=False)
                else:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def generate_summary(self):
        case = self.data[self.case_selector.currentIndex()]
        combined_text = f"Suspect: {case['suspect_statement']}\n\n"
        combined_text += "\n".join([f"Witness {i + 1}: {w}" for i, w in enumerate(case['witness_statements'])])
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            summary = summarizer(combined_text[:1000], max_length=120, min_length=40, do_sample=False)
            self.summary_text.setText(summary[0]['summary_text'])
            self.generated_summary = summary[0]['summary_text']
        except Exception as e:
            self.summary_text.setText("Summary generation failed.\n" + str(e))
            self.generated_summary = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InterrogationAnalyzer()
    window.show()
    sys.exit(app.exec())
