from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from rag_chain import process_query_with_mode, add_document_from_path

app = Flask(__name__)
# Enable CORS so frontend can communicate with backend if hosted separately
CORS(app) 

app.config["UPLOAD_FOLDER"] = "uploaded_files"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    user_input = data.get("question", "").strip()
    mode = data.get("mode", "hybrid") # Defaults to hybrid if missing
    
    if not user_input:
        return jsonify({"answer": "🤖 Please ask a question."})

    try:
        # Pass the mode into your processing function
        answer = process_query_with_mode(user_input, mode) 
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"🤖 Error: {str(e)}"})

@app.route("/upload", methods=["POST"])
def upload():
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
    # Disable the reloader so PyTorch cache files don't trigger a server restart
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
