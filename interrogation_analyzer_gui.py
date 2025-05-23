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
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from statement_similarity_analysis import (analyze_statements, extract_key_points, 
                                         compare_key_points, calculate_confidence_score,
                                         generate_detailed_explanation, create_visualizations)

# Lazy loading of models
class ModelLoader:
    _instance = None
    _similarity_model = None
    _nlp = None

    @classmethod
    def get_similarity_model(cls):
        if cls._similarity_model is None:
            cls._similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return cls._similarity_model

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load("en_core_web_sm")
        return cls._nlp

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
        
        # Batch process witness statements
        witness_statements = self.case['witness_statements']
        suspect_statement = self.case['suspect_statement']
        
        # Process all witness statements
        for i, witness in enumerate(witness_statements):
            analysis = analyze_statements(suspect_statement, witness)
            results[f'witness_{i+1}'] = {
                'statement': witness,
                'similarity': analysis['similarity'],
                'confidence': analysis['confidence'],
                'comparison': analysis['comparison'],
                'explanation': generate_detailed_explanation(analysis, suspect_statement, witness),
                'result': "✅ Consistent/Supportive" if analysis['similarity'] > 0.6 and analysis['confidence'] > 0.5 else 
                         "❌ Contradictory" if analysis['similarity'] < 0.3 or analysis['confidence'] < 0.3 else 
                         "⚠️ Unclear/Neutral"
            }
            current_step += 1
            self.progress.emit(int(current_step * 100 / total_steps))
        
        # Extract keywords efficiently
        all_text = suspect_statement + " " + " ".join(witness_statements)
        doc = self.nlp(all_text)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]
        results['keywords'] = [word for word, count in Counter(keywords).most_common(10)]
        
        # Create visualizations in background
        create_visualizations([results[f'witness_{i+1}'] for i in range(len(witness_statements))], 
                            self.case['description'])
        
        current_step += 1
        self.progress.emit(int(current_step * 100 / total_steps))
        
        self.finished.emit(results)

class InterrogationAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interrogation Statement Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load data
        with open("forensic_statements_data_en.json", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.init_ui()
        
        # Initialize with first case
        self.update_analysis(0)
    
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Top controls
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Suspect statement
        left_layout.addWidget(QLabel("Suspect Statement:"))
        self.suspect_text = QTextEdit()
        self.suspect_text.setReadOnly(True)
        left_layout.addWidget(self.suspect_text)
        
        # Summary section
        left_layout.addWidget(QLabel("Summary:"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        left_layout.addWidget(self.summary_text)
        
        # Summary button
        self.summary_btn = QPushButton("Generate Summary")
        self.summary_btn.clicked.connect(self.generate_summary)
        left_layout.addWidget(self.summary_btn)
        
        content_splitter.addWidget(left_panel)
        
        # Right panel with tabs
        right_panel = QTabWidget()
        
        # Analysis tab
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Witness statements
        analysis_layout.addWidget(QLabel("Witness Statements:"))
        self.witness_text = QTextEdit()
        self.witness_text.setReadOnly(True)
        analysis_layout.addWidget(self.witness_text)
        
        # Analysis results
        analysis_layout.addWidget(QLabel("Analysis Results:"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        right_panel.addTab(analysis_widget, "Analysis")
        
        # Timeline tab
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        timeline_layout.addWidget(self.canvas)
        right_panel.addTab(timeline_widget, "Timeline")
        
        # Keywords tab
        keywords_widget = QWidget()
        keywords_layout = QVBoxLayout(keywords_widget)
        self.keywords_text = QTextEdit()
        self.keywords_text.setReadOnly(True)
        keywords_layout.addWidget(self.keywords_text)
        right_panel.addTab(keywords_widget, "Keywords")
        
        # Add panels to splitter
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([400, 1000])
        
        layout.addWidget(content_splitter)
    
    def update_analysis(self, index):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        case = self.data[index]
        self.suspect_text.setText(case['suspect_statement'])
        
        # Create and start worker thread
        self.worker = AnalysisWorker(case, ModelLoader.get_similarity_model(), ModelLoader.get_nlp())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.update_ui_with_results)
        self.worker.start()
    
    def update_ui_with_results(self, results):
        # Update witness statements and analysis
        witness_text = ""
        analysis_text = ""
        
        for i in range(1, len(results)):
            if f'witness_{i}' in results:
                witness_text += f"Witness {i}:\n{results[f'witness_{i}']['statement']}\n\n"
                analysis_text += f"Witness {i} Analysis:\n"
                analysis_text += f"Similarity Score: {results[f'witness_{i}']['similarity']:.2f}\n"
                analysis_text += f"Confidence Score: {results[f'witness_{i}']['confidence']:.2f}\n"
                analysis_text += f"Result: {results[f'witness_{i}']['result']}\n\n"
                analysis_text += "Detailed Explanation:\n"
                analysis_text += results[f'witness_{i}']['explanation'] + "\n\n"
        
        self.witness_text.setText(witness_text)
        self.analysis_text.setText(analysis_text)
        
        # Update keywords
        if 'keywords' in results:
            self.keywords_text.setText("\n".join(results['keywords']))
        
        # Update timeline
        self.update_timeline(results)
        
        # Update summary
        if 'summary' in results:
            self.summary_text.setText(results['summary'])
        
        self.progress_bar.setVisible(False)
    
    def update_timeline(self, results):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create timeline data more efficiently
        statements = ['Suspect'] + [f'Witness {i+1}' for i in range(len(results)-2)]
        similarities = [1.0] + [results[f'witness_{i+1}']['similarity'] for i in range(len(results)-2)]
        
        # Optimize plotting
        ax.plot(range(len(statements)), similarities, 'bo-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(statements)))
        ax.set_xticklabels(statements, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Statement Similarity Timeline')
        
        # Optimize layout
        self.figure.tight_layout()
        self.canvas.draw()
    
    def export_analysis(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if file_name:
            case = self.data[self.case_selector.currentIndex()]
            analysis_data = {
                'case_description': case['description'],
                'suspect_statement': case['suspect_statement'],
                'witness_statements': case['witness_statements'],
                'analysis': self.analysis_text.toPlainText(),
                'keywords': self.keywords_text.toPlainText(),
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                if file_name.endswith('.csv'):
                    df = pd.DataFrame([analysis_data])
                    df.to_csv(file_name, index=False)
                else:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export analysis: {str(e)}")
    
    def generate_summary(self):
        case = self.data[self.case_selector.currentIndex()]
        combined_text = f"Suspect: {case['suspect_statement']}\n\n"
        combined_text += "\n".join([f"Witness {i+1}: {w}" for i, w in enumerate(case['witness_statements'])])
        
        summary = ModelLoader.get_similarity_model().encode(combined_text).tolist()
        results = {
            'summary': summary,
            'keywords': [word for word, count in Counter([token.text for token in ModelLoader.get_nlp(combined_text) if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]).most_common(10)]
        }
        
        self.update_ui_with_results(results)
    
    def generate_contradiction(self, statement1, statement2):
        return f"Key contradiction points:\n" + \
               f"1. Statement 1: {statement1[:100]}...\n" + \
               f"2. Statement 2: {statement2[:100]}...\n" + \
               f"These statements show significant differences in their accounts."

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InterrogationAnalyzer()
    window.show()
    sys.exit(app.exec()) 