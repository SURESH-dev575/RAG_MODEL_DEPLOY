from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from rag_chain import hybrid_rag, add_document_from_path  # local module

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploaded_files"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    user_input = data.get("question", "").strip()
    if not user_input:
        return jsonify({"answer": "🤖 Please ask a question."})

    try:
        answer = hybrid_rag(user_input)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"🤖 Error: {str(e)}"})

@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts file uploads from the frontend, saves them, and adds to vector DB via rag_chain.add_document_from_path.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "msg": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Add to vector DB
    try:
        add_document_from_path(save_path)
        return jsonify({"ok": True, "msg": f"'{filename}' uploaded and indexed."})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Indexing error: {str(e)}"}), 500

if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
