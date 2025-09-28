import gradio as gr
import google.generativeai as genai
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "medical_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "your_password")
    )

model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Get doctors from database
def get_doctors_by_specialty(specialty):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT name, specialty, phone, location, experience 
        FROM doctors 
        WHERE LOWER(specialty) LIKE %s 
        ORDER BY experience DESC 
        LIMIT 3
        """
        
        cursor.execute(query, (f'%{specialty.lower()}%',))
        doctors = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return doctors
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Main medical bot function
def medical_bot(message, history):
    try:
        # Create chat with system instruction as first message
        chat = model.start_chat(history=[])
        
        # Send system instruction + user message together
        full_prompt = f"""
        You are a medical assistant. Based on user's health problem, suggest which type of doctor they should visit.

        Common specialties:
        - Fever/Cold/Cough → General Physician
        - Heart problems → Cardiologist  
        - Skin problems → Dermatologist
        - Kidney problems → Nephrologist
        - Stomach/Digestive → Gastroenterologist
        - Bone/Joint pain → Orthopedic
        - Women's health → Gynecologist
        - Brain/Nerve → Neurologist
        - Eye problems → Ophthalmologist
        - Ear/Nose/Throat → ENT

        Always respond in this format:
        PROBLEM: [user's problem]
        DOCTOR: [specialty name]
        REASON: [why this doctor]

        User's question: {message}
        """
        
        ai_response = chat.send_message(full_prompt)
        response_text = ai_response.text
        
        # Extract specialty from AI response
        specialty = None
        if "DOCTOR:" in response_text:
            doctor_line = [line for line in response_text.split('\n') if 'DOCTOR:' in line]
            if doctor_line:
                specialty = doctor_line[0].replace('DOCTOR:', '').strip()
        
        # Get doctors if specialty found
        doctor_suggestions = ""
        if specialty:
            doctors = get_doctors_by_specialty(specialty)
            if doctors:
                doctor_suggestions = f"\n\n **Recommended Doctors ({specialty}):**\n"
                for i, doctor in enumerate(doctors, 1):
                    name, spec, phone, location, experience = doctor
                    doctor_suggestions += f"{i}. **{name}** - {spec}\n"
                    doctor_suggestions += f"   📞 {phone} | 📍 {location} | {experience} years experience\n\n"
        
        # Combine AI response with doctor suggestions
        final_response = response_text + doctor_suggestions
        
        # Add disclaimer
        final_response += "\n **Please consult with a qualified doctor for proper diagnosis.**"
        
        return final_response
        
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
demo = gr.ChatInterface(
    fn=medical_bot,
    title="Medical Bot with Doctor Finder",
    description="Ask about health problems and get doctor recommendations",
    examples=[
        "আমার জ্বর হয়েছে",  # Fever
        "কিডনিতে সমস্যা",     # Kidney problem  
        "বুকে ব্যথা",        # Chest pain
        "পেটে ব্যথা",        # Stomach pain
        "চামড়ায় র‍্যাশ"      # Skin rash
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()