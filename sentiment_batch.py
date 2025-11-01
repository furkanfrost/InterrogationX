import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from statement_similarity_analysis import analyze_statements
from emotion_sentiment_analyzer import analyze_emotion_sentiment

DATA_FILE = "forensic_statements_data_en.json"
OUTPUT_DIR = "emotion_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_all_cases():
    with open(DATA_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    all_results = []

    for case in cases:
        case_id = case["case_id"]
        desc = case["description"]
        suspect = case["suspect_statement"]
        witnesses = case["witness_statements"]

        print(f"\n🧾 Analyzing case {case_id}: {desc}")
        suspect_em = analyze_emotion_sentiment(suspect)

        for i, witness in enumerate(witnesses, start=1):
            base = analyze_statements(suspect, witness)
            witness_em = analyze_emotion_sentiment(witness)

            result = {
                "case_id": case_id,
                "description": desc,
                "witness_index": i,
                "suspect_statement": suspect,
                "witness_statement": witness,
                "similarity": base["similarity"],
                "confidence": base["confidence"],
                "suspect_sentiment": suspect_em["sentiment"],
                "suspect_emotion": suspect_em["emotion"],
                "witness_sentiment": witness_em["sentiment"],
                "witness_emotion": witness_em["emotion"],
                "suspect_sentiment_score": suspect_em["sentiment_score"],
                "witness_sentiment_score": witness_em["sentiment_score"],
                "suspect_emotion_score": suspect_em["emotion_score"],
                "witness_emotion_score": witness_em["emotion_score"],
            }

            all_results.append(result)

    save_results(all_results)
    visualize_emotions(all_results)
    print(f"\n✅ Done! Results saved in '{OUTPUT_DIR}' folder.")

def save_results(results):
    json_path = os.path.join(OUTPUT_DIR, "emotion_sentiment_results.json")
    csv_path = os.path.join(OUTPUT_DIR, "emotion_sentiment_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "results": results}, f, indent=4, ensure_ascii=False)

    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"💾 Saved: {json_path}")
    print(f"💾 Saved: {csv_path}")

def visualize_emotions(results):
    # Witness emotion distribution
    emotions = [r["witness_emotion"] for r in results]
    counts = Counter(emotions)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title("Witness Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "emotion_distribution.png"))
    plt.close()
    similarities = [r["similarity"] for r in results]
    emo_scores = [r["witness_emotion_score"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.scatter(similarities, emo_scores, alpha=0.7)
    plt.title("Similarity vs Witness Emotion Score")
    plt.xlabel("Similarity")
    plt.ylabel("Emotion Confidence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "similarity_vs_emotion.png"))
    plt.close()

if __name__ == "__main__":
    analyze_all_cases()
