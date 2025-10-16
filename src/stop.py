"""
Sex Chat Detection using User-Word Bipartite Graph

Author: Ramaguru Radhakrishnan
Date: 16th October 2025

This script:
1. Loads a JSON lexicon with word severities.
2. Processes chat messages between users.
3. Builds a bipartite graph connecting users to words they used.
4. Computes a User Risk Index (URI) based on word severities.
5. Flags conversations as "Sex Chat Detected" or "Not Detected".
6. Optionally visualizes the bipartite graph with risk heatmap.
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json

# -------------------------------
# Step 1: Load Lexicon from JSON
# -------------------------------
# The JSON should have entries like:
# [{"word": "example", "severity": "high"}, ...]
with open("data.json") as f:
    lexicon_data = json.load(f)

# Create mapping: word -> severity
LEXICON = {entry["word"]: entry["severity"] for entry in lexicon_data}
SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}

# -------------------------------
# Step 2: Example Chat Dataset
# -------------------------------
# Each entry: (sender, receiver, message)
chats = [
    ("U1", "U2", "i will lick your pussy"),
    ("U1", "U2", "is that ok?"),
    ("U1", "U2", "yes, please do it"),
    ("U2", "U1", "i love chatting with you"),
    ("U2", "U1", "good night sweetheart"),
    ("U2", "U1", "send me a kiss emoji"),
    ("U3", "U4", "what is the assignment question"),
    ("U3", "U4", "we will study tomorrow")
]

# -------------------------------
# Step 3: Tokenizer Function
# -------------------------------
def tokenize(text):
    """
    Tokenize input text into lowercase words (alphanumeric only).
    """
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

# -------------------------------
# Step 4: Build Bipartite Graph
# -------------------------------
B = nx.Graph()
user_nodes = set()
word_nodes = set()

for sender, receiver, message in chats:
    words = tokenize(message)
    for user in [sender, receiver]:
        user_nodes.add(user)
        for word in words:
            word_nodes.add(word)
            weight = SEVERITY_WEIGHT.get(LEXICON.get(word, "low"), 1)
            if B.has_edge(user, word):
                B[user][word]["weight"] += weight
            else:
                B.add_edge(user, word, weight=weight)

# -------------------------------
# Step 5: Compute User Risk Index (URI)
# -------------------------------
user_risk = {}

for user in user_nodes:
    edges = B.edges(user, data=True)
    if not edges:
        continue
    # Weighted degree: sum of severity weights for user's words
    total_weight = sum(d["weight"] for _, _, d in edges)
    # Influence by number of connected words (normalized)
    connected_words = len(edges)
    uri = total_weight * (connected_words / max(1, len(word_nodes)))
    user_risk[user] = round(uri, 2)

# -------------------------------
# Step 6: Conversation Detection
# -------------------------------
# Aggregate risk across all users to flag conversation
conversation_risk = sum(user_risk.values()) / len(user_risk)
THRESHOLD = 10  # Threshold for "Sex Chat Detected"
status = "Sex Chat Detected" if conversation_risk >= THRESHOLD else "Not Detected"

print("🚨 Conversation Detection Result")
print(f"Conversation Risk Index: {conversation_risk}")
print(f"Status: {status}\n")

# Output URI per user
print("🧑 User Risk Index (URI) per user:")
for user, score in sorted(user_risk.items(), key=lambda x: x[1], reverse=True):
    print(f"{user}: URI = {score}")

# -------------------------------
# Step 7: Top Contributing Words (Explainability)
# -------------------------------
print("\n📝 Top contributing words per user:")
for user in sorted(user_risk, key=user_risk.get, reverse=True):
    edges = B.edges(user, data=True)
    sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    top_words = [f"{w}({d['weight']})" for _, w, d in sorted_edges[:5]]  # top 5
    print(f"{user}: {', '.join(top_words)}")

# -------------------------------
# Step 8: Optional Graph Visualization
# -------------------------------
# Compute positions for visualization
pos = nx.spring_layout(B, k=0.8)

# Node colors: users red (opacity = risk), words lightblue
node_colors = []
for n in B.nodes():
    if n in user_nodes:
        risk = user_risk.get(n, 0)
        alpha = min(0.2 + risk / 50, 1)  # scale opacity
        node_colors.append((1, 0, 0, alpha))
    else:
        node_colors.append('lightblue')

plt.figure(figsize=(10, 8))
nx.draw(B, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10,
        width=[B[u][v]['weight'] for u,v in B.edges()])  # edge thickness by weight
plt.title("User-Word Bipartite Graph with Risk Heatmap")
plt.show()

