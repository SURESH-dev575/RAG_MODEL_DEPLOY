from flask import Flask, render_template, request, jsonify
from rag_chain import hybrid_rag

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"answer": "🤖 Please ask a question."})
    try:
        response = hybrid_rag(user_input)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"answer": f"🤖 Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
