from flask import Flask, jsonify, request
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
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not db_type or db_type not in ['graph', 'vector']:
        return jsonify({"error": "Invalid or missing db_choice"}), 400

    try:
        # Create an instance of the Chat class based on db_type
        chat_instance = Chat(db_type=db_type)
        # Query the database
        answer = chat_instance.query_db(query)
        return jsonify(answer)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
