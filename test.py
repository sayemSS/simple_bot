from flask import Flask,request ,jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError('GEMINI_API_KEY variable not set')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

chat = model.start_chat(history=[])

app = Flask(__name__)

@app.route('/chat',methods=['POST'])

def chat_bot():
    try:
        data = request.get_json()

        if not data or 'data' not in data or 'message' not in data['data']:
            return jsonify ({"error" : 'invalilid request format'}),400
        
        user_id = data['data'].get('id')
        user_messege = data['data'].get('id')

        response = chat.send_message(user_messege)

        return jsonify({
            'id': user_id,
            "user_message" : user_messege,
            "bot_response" : response.text
        })
    except Exception as e:
        return jsonify({'error': str(e)}),500
    

    if __name__ == "__main__":
        import os 
        port = int(os.environ.get("PORT",5000))
  

