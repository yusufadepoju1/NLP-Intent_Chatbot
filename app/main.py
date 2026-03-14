from flask import Flask, render_template, request, jsonify
from app.chatbot import get_response

app = Flask(__name__, template_folder="../templates", static_folder="../static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("message")
    if not msg:
        return jsonify({"response": "Empty message"}), 400
    
    response = get_response(msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)