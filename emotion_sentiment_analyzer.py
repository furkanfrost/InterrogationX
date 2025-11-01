from transformers import pipeline

_sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
_emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=False)

def analyze_emotion_sentiment(text: str) -> dict:
    try:
        sentiment_result = _sentiment_model(text[:512])[0]
        emotion_result = _emotion_model(text[:512])[0]

        return {
            "sentiment": sentiment_result["label"],
            "sentiment_score": sentiment_result["score"],
            "emotion": emotion_result["label"],
            "emotion_score": emotion_result["score"]
        }
    except Exception as e:
        return {
            "sentiment": "unknown",
            "sentiment_score": 0.0,
            "emotion": "unknown",
            "emotion_score": 0.0,
            "error": str(e)
        }
