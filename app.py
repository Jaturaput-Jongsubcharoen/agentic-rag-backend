# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_graph import rag_answer   

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    question = data.get("message")
    if not question:
        return jsonify({"error": "Missing 'message' field"}), 400

    try:
        result = rag_answer(question)

        return jsonify({
            "answer": result["answer"],
            "context": result["context"],
            "sources": result["sources"],
            "history": result["history"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
