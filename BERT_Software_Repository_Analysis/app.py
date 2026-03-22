from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from utils.bert_embed import get_embedding

app = Flask(__name__)

df = pd.read_csv("data/repository_text.csv")

embeddings = np.vstack(df["text"].apply(get_embedding))

kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)

cluster_counts = df["cluster"].value_counts().to_dict()

cluster_names = {
    0: "UI Bug",
    1: "Performance Bug",
    2: "Logic / Functional Bug",
    3: "Authentication Bug",
    4: "Database Bug"
}

solution_map = {
    "UI Bug": "Check frontend event handlers, inspect CSS alignment, and test across browsers and devices.",
    "Performance Bug": "Profile execution, optimize algorithms, analyze logs, and improve database query performance.",
    "Logic / Functional Bug": "Review business logic, validate edge cases, add unit tests, and debug conditional statements.",
    "Authentication Bug": "Verify login flow, token handling, session management, and access control rules.",
    "Database Bug": "Check database connectivity, optimize queries, add indexing, and monitor database load."
}

# Severity logic
def assign_severity(count):
    if count >= 500:
        return "High Severity"
    elif count >= 200:
        return "Medium Severity"
    else:
        return "Low Severity"

def calculate_confidence(similarity):
    return round(float(similarity) * 100, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]

        new_embedding = get_embedding(text)
        similarity = cosine_similarity(new_embedding, embeddings)[0]
        idx = np.argmax(similarity)
        max_similarity = similarity[idx]

        cluster_id = df.iloc[idx]["cluster"]
        cluster_type = cluster_names.get(cluster_id, "General Bug")

        count = cluster_counts.get(cluster_id, 1)
        severity = assign_severity(count)
        solution = solution_map.get(
            cluster_type,
            "Analyze logs and debug the issue manually."
        )
        
        confidence = calculate_confidence(max_similarity)

        return render_template(
            "result.html",
            text=text,
            cluster=cluster_type,
            severity=severity,
            count=count,
            solution=solution,
            confidence=confidence
        )

    return render_template("index.html")


@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html')


@app.route('/root-files/<path:filename>')
def root_files(filename):
    return send_from_directory(os.getcwd(), filename)


@app.route('/performance')
def performance_page():
    return render_template('performance.html')

if __name__ == "__main__":
    app.run(debug=True)
