from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS
import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")


genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')


chat = model.start_chat(history=[])


app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    try:
        data = request.get_json()  # Parse incoming JSON

        if not data or 'data' not in data or 'message' not in data['data']:
            return jsonify({"error": "Invalid request format"}), 400

        user_id = data['data'].get('id')
        user_message = data['data'].get('message', "")

        # Send user message to Gemini
        response = chat.send_message(user_message)

        return jsonify({
            "id": user_id,
            "user_message": user_message,
            "bot_response": response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
