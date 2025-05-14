
from sentence_transformers import SentenceTransformer, util
import json

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load data
with open("forensic_statements_data_en.json", encoding="utf-8") as f:
    data = json.load(f)

# Compare suspect and witness statements
for case in data:
    suspect = case["suspect_statement"]
    witnesses = case["witness_statements"]
    
    print(f"\n📁 Case: {case['description']}")
    print(f"👤 Suspect: {suspect}")
    
    for i, witness in enumerate(witnesses):
        similarity = util.cos_sim(
            model.encode(suspect, convert_to_tensor=True),
            model.encode(witness, convert_to_tensor=True)
        ).item()
        
        print(f"🧾 Witness {i+1}: {witness}")
        print(f"   🔍 Similarity Score: {similarity:.2f}")
        
        if similarity > 0.6:
            result = "✅ Consistent/Supportive"
        elif similarity < 0.3:
            result = "❌ Contradictory"
        else:
            result = "⚠️ Unclear/Neutral"
        
        print(f"   🧠 Analysis: {result}")
