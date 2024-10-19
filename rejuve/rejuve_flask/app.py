# app.py
from flask import Flask, jsonify, request
import asyncio
from chatbot import Chat  # Import the Chat class

app = Flask(__name__)


@app.route('/')
def home():
    data = {
        "message": "Welcome to the Flask API!",
        "status": "success"
    }
    return jsonify(data)


@app.route('/chat', methods=['POST'])
async def chat():
    query = request.json.get('query')
    db_type = request.json.get('db_choice')
    # Create an instance of the Chat class
    chat_instance = Chat(db_type=db_type)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        answer = await chat_instance.query_db(query, )
        return answer
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
