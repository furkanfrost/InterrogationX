from sentence_transformers import SentenceTransformer, util
import json
import spacy
from collections import defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np
from datetime import datetime

# Load models
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_key_points(text):
    doc = nlp(text)
    key_points = {
        'time': [],
        'location': [],
        'actions': [],
        'entities': [],
        'modifiers': [],
        'sentiment': [],
        'temporal': [],
        'certainty': [],
        'context': []
    }
    
    # Extract named entities and their types
    for ent in doc.ents:
        if ent.label_ in ['TIME', 'DATE']:
            key_points['time'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC', 'FAC']:
            key_points['location'].append(ent.text)
        elif ent.label_ in ['PERSON', 'ORG']:
            key_points['entities'].append(ent.text)
    
    # Extract verbs, adjectives, and adverbs
    for token in doc:
        if token.pos_ == 'VERB':
            key_points['actions'].append(token.text)
            # Check for temporal indicators
            if token.dep_ in ['aux', 'auxpass']:
                key_points['temporal'].append(token.text)
        elif token.pos_ in ['ADJ', 'ADV']:
            key_points['modifiers'].append(token.text)
            # Check for certainty indicators
            if token.text.lower() in ['definitely', 'certainly', 'maybe', 'possibly', 'probably']:
                key_points['certainty'].append(token.text)
    
    # Basic sentiment analysis
    sentiment_score = 0
    for token in doc:
        if token.pos_ in ['ADJ', 'ADV']:
            if token.text.lower() in ['good', 'great', 'excellent', 'positive']:
                sentiment_score += 1
            elif token.text.lower() in ['bad', 'terrible', 'negative', 'awful']:
                sentiment_score -= 1
    
    key_points['sentiment'].append('positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral')
    
    # Extract context clues
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in ['nsubj', 'dobj']:
            key_points['context'].append(chunk.text)
    
    return key_points

def compare_key_points(suspect_points, witness_points):
    comparison = {
        'time': {'matching': [], 'different': []},
        'location': {'matching': [], 'different': []},
        'actions': {'matching': [], 'different': []},
        'entities': {'matching': [], 'different': []},
        'modifiers': {'matching': [], 'different': []},
        'sentiment': {'matching': [], 'different': []},
        'temporal': {'matching': [], 'different': []},
        'certainty': {'matching': [], 'different': []},
        'context': {'matching': [], 'different': []}
    }
    
    for category in comparison:
        suspect_items = set(suspect_points[category])
        witness_items = set(witness_points[category])
        
        comparison[category]['matching'] = list(suspect_items.intersection(witness_items))
        comparison[category]['different'] = list(suspect_items.symmetric_difference(witness_items))
    
    return comparison

def calculate_confidence_score(comparison):
    total_points = 0
    matching_points = 0
    
    for category in comparison:
        total_points += len(comparison[category]['matching']) + len(comparison[category]['different'])
        matching_points += len(comparison[category]['matching'])
    
    if total_points == 0:
        return 0.0
    
    return matching_points / total_points

def generate_statistics(analysis_results):
    stats = {
        'total_cases': len(analysis_results),
        'average_similarity': np.mean([r['similarity'] for r in analysis_results]),
        'average_confidence': np.mean([r['confidence'] for r in analysis_results]),
        'consistency_distribution': defaultdict(int)
    }
    
    for result in analysis_results:
        if result['similarity'] > 0.6 and result['confidence'] > 0.5:
            stats['consistency_distribution']['Strongly Consistent'] += 1
        elif result['similarity'] > 0.4 and result['confidence'] > 0.3:
            stats['consistency_distribution']['Partially Consistent'] += 1
        else:
            stats['consistency_distribution']['Contradictory'] += 1
    
    return stats

def create_visualizations(analysis_results, case_id):
    # Create directory for visualizations if it doesn't exist
    import os
    if not os.path.exists('analysis_visualizations'):
        os.makedirs('analysis_visualizations')
    
    # Similarity vs Confidence scatter plot
    plt.figure(figsize=(10, 6))
    similarities = [r['similarity'] for r in analysis_results]
    confidences = [r['confidence'] for r in analysis_results]
    plt.scatter(similarities, confidences, alpha=0.6)
    plt.xlabel('Similarity Score')
    plt.ylabel('Confidence Score')
    plt.title('Statement Analysis: Similarity vs Confidence')
    plt.grid(True)
    plt.savefig(f'analysis_visualizations/case_{case_id}_scatter.png')
    plt.close()
    
    # Category comparison bar chart
    categories = list(analysis_results[0]['comparison'].keys())
    matching_counts = []
    different_counts = []
    
    for category in categories:
        matching = sum(len(r['comparison'][category]['matching']) for r in analysis_results)
        different = sum(len(r['comparison'][category]['different']) for r in analysis_results)
        matching_counts.append(matching)
        different_counts.append(different)
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, matching_counts, width, label='Matching')
    plt.bar(x + width/2, different_counts, width, label='Different')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Statement Analysis: Category Comparison')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'analysis_visualizations/case_{case_id}_categories.png')
    plt.close()

def generate_detailed_explanation(analysis, suspect, witness):
    explanation = []
    
    # Overall assessment
    if analysis['similarity'] > 0.6 and analysis['confidence'] > 0.5:
        explanation.append("The statements show strong consistency in their accounts.")
        explanation.append("Key points align well, and there are minimal contradictions in the details provided.")
    elif analysis['similarity'] > 0.4 and analysis['confidence'] > 0.3:
        explanation.append("The statements show partial consistency.")
        explanation.append("While some key points align, there are notable differences that require further investigation.")
    else:
        explanation.append("The statements show significant contradictions.")
        explanation.append("Key elements of the accounts differ substantially, suggesting potential inconsistencies in the testimony.")
    
    # Detailed analysis by category
    explanation.append("\nDetailed Analysis by Category:")
    
    for category, details in analysis['comparison'].items():
        if details['matching'] or details['different']:
            explanation.append(f"\n{category.upper()}:")
            if details['matching']:
                explanation.append(f"  • Matching elements: {', '.join(details['matching'])}")
            if details['different']:
                explanation.append(f"  • Different elements: {', '.join(details['different'])}")
    
    # Specific contradictions
    contradictions = []
    for category, details in analysis['comparison'].items():
        if details['different']:
            contradictions.append(f"  • {category.title()}: {', '.join(details['different'])}")
    
    if contradictions:
        explanation.append("\nKey Contradictions:")
        explanation.extend(contradictions)
    
    # Reliability assessment
    explanation.append("\nReliability Assessment:")
    if analysis['confidence'] > 0.5:
        explanation.append("  • High confidence in the analysis")
    elif analysis['confidence'] > 0.3:
        explanation.append("  • Moderate confidence in the analysis")
    else:
        explanation.append("  • Low confidence in the analysis")
    
    return "\n".join(explanation)

def analyze_new_statements(suspect_statement, witness_statements, case_description=""):
    """
    Analyze new statements provided by the user.
    
    Args:
        suspect_statement (str): The suspect's statement
        witness_statements (list): List of witness statements
        case_description (str): Optional description of the case
    
    Returns:
        dict: Analysis results including detailed explanations
    """
    results = {
        'case_description': case_description,
        'suspect_statement': suspect_statement,
        'witness_analyses': []
    }
    
    for i, witness in enumerate(witness_statements):
        analysis = analyze_statements(suspect_statement, witness)
        analysis['witness_statement'] = witness
        analysis['witness_number'] = i + 1
        analysis['detailed_explanation'] = generate_detailed_explanation(
            analysis, suspect_statement, witness
        )
        results['witness_analyses'].append(analysis)
    
    return results

def print_analysis_results(results):
    """Print the analysis results in a formatted way."""
    print(f"\n{'='*100}")
    if results['case_description']:
        print(f"📁 Case: {results['case_description']}")
    print(f"👤 Suspect Statement: {results['suspect_statement']}")
    print(f"{'='*100}")
    
    # Create analysis table
    table_data = []
    for analysis in results['witness_analyses']:
        table_data.append([
            f"Witness {analysis['witness_number']}",
            f"{analysis['similarity']:.2f}",
            f"{analysis['confidence']:.2f}",
            "✅ Strongly Consistent" if analysis['similarity'] > 0.6 and analysis['confidence'] > 0.5
            else "⚠️ Partially Consistent" if analysis['similarity'] > 0.4 and analysis['confidence'] > 0.3
            else "❌ Contradictory"
        ])
    
    print("\n📊 Analysis Summary Table:")
    print(tabulate(table_data, 
                  headers=['Witness', 'Similarity', 'Confidence', 'Assessment'],
                  tablefmt='grid'))
    
    for analysis in results['witness_analyses']:
        print(f"\n{'='*100}")
        print(f"🧾 Witness {analysis['witness_number']} Statement: {analysis['witness_statement']}")
        print(f"{'='*100}")
        
        print(f"\n📊 Analysis Results:")
        print(f"   🔍 Overall Similarity Score: {analysis['similarity']:.2f}")
        print(f"   🎯 Confidence Score: {analysis['confidence']:.2f}")
        
        print("\n📝 Detailed Explanation:")
        print(analysis['detailed_explanation'])
        
        print(f"\n{'-'*100}")

def analyze_statements(suspect, witness):
    # Calculate overall similarity
    similarity = util.cos_sim(
        similarity_model.encode(suspect, convert_to_tensor=True),
        similarity_model.encode(witness, convert_to_tensor=True)
    ).item()
    
    # Extract and compare key points
    suspect_points = extract_key_points(suspect)
    witness_points = extract_key_points(witness)
    comparison = compare_key_points(suspect_points, witness_points)
    
    # Calculate confidence score
    confidence = calculate_confidence_score(comparison)
    
    return {
        'similarity': similarity,
        'comparison': comparison,
        'confidence': confidence
    }

# Example usage of the new function
if __name__ == "__main__":
    # Load and analyze existing data
    with open("forensic_statements_data_en.json", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process all cases
    all_analysis_results = []
    for case in data:
        case_analysis = []
        for witness in case['witness_statements']:
            analysis = analyze_statements(case['suspect_statement'], witness)
            case_analysis.append(analysis)
        all_analysis_results.append(case_analysis)
        
        # Print detailed analysis for this case
        print_detailed_analysis(case, case_analysis)
        
        # Create visualizations
        create_visualizations(case_analysis, case['case_id'])
    
    # Generate and print overall statistics
    stats = generate_statistics([item for sublist in all_analysis_results for item in sublist])
    print("\n📈 Overall Analysis Statistics:")
    print(f"Total Cases Analyzed: {stats['total_cases']}")
    print(f"Average Similarity Score: {stats['average_similarity']:.2f}")
    print(f"Average Confidence Score: {stats['average_confidence']:.2f}")
    print("\nConsistency Distribution:")
    for category, count in stats['consistency_distribution'].items():
        print(f"{category}: {count} cases ({(count/stats['total_cases']*100):.1f}%)")
    
    # Example of analyzing new statements
    print("\n\nExample of analyzing new statements:")
    new_case = {
        'description': "New case example",
        'suspect_statement': "I was at the store at 2 PM and bought some groceries.",
        'witness_statements': [
            "I saw him at the store around 2 PM buying food.",
            "He wasn't at the store, I was there all day."
        ]
    }
    
    results = analyze_new_statements(
        new_case['suspect_statement'],
        new_case['witness_statements'],
        new_case['description']
    )
    print_analysis_results(results)
