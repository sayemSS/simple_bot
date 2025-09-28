import gradio as gr
import google.generativeai as genai
import psycopg2
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pickle

load_dotenv()

# Configure API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize LangChain
llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

class SimpleMedicalBot:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.setup_rag()
    
    def get_db_connection(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "medical_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "your_password")
        )
    
    def load_doctors_from_db(self):
        """Load all doctors from database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = "SELECT name, specialty, phone, location, experience FROM doctors"
            cursor.execute(query)
            doctors = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return doctors
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
    def create_doctor_documents(self):
        """Convert database records to LangChain documents"""
        doctors = self.load_doctors_from_db()
        documents = []
        
        for doctor in doctors:
            name, specialty, phone, location, experience = doctor
            
            # Create document content
            content = f"""
            Doctor: {name}
            Specialty: {specialty}
            Phone: {phone}
            Location: {location}
            Experience: {experience} years
            
            This doctor specializes in {specialty} and is located in {location}.
            """
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "name": name,
                    "specialty": specialty,
                    "phone": phone,
                    "location": location,
                    "experience": experience
                }
            )
            documents.append(doc)
        
        return documents
    
    def setup_rag(self):
        """Setup RAG system with database data"""
        try:
            # Try loading existing vector store
            if os.path.exists("medical_vectorstore.pkl"):
                with open("medical_vectorstore.pkl", "rb") as f:
                    self.vector_store = pickle.load(f)
                print("✅ Loaded existing vector store")
            else:
                # Create new vector store from database
                documents = self.create_doctor_documents()
                if documents:
                    self.vector_store = FAISS.from_documents(documents, embeddings)
                    
                    # Save vector store
                    with open("medical_vectorstore.pkl", "wb") as f:
                        pickle.dump(self.vector_store, f)
                    print("✅ Created new vector store from database")
                else:
                    print("❌ No documents found in database")
                    return
            
            # Create QA chain
            self.create_qa_chain()
            
        except Exception as e:
            print(f"❌ Error setting up RAG: {e}")
    
    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        
        prompt_template = """
        You are a medical assistant. Based on the doctor information from the database, help users find the right specialist.

        Database Information:
        {context}

        User Question: {question}

        Respond in this format:
        PROBLEM: [user's health issue]
        SPECIALIST: [recommended doctor type]
        REASON: [why this specialist is needed]

        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def get_doctors_by_specialty(self, specialty):
        """Get doctors from database by specialty"""
        try:
            conn = self.get_db_connection()
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
    
    def process_query(self, message, history):
        """Main function to process user queries"""
        try:
            if not self.qa_chain:
                return "❌ RAG system not initialized. Please check database connection."
            
            # Get RAG response
            rag_response = self.qa_chain.run(message)
            
            # Extract specialist type from response
            specialist = None
            if "SPECIALIST:" in rag_response:
                lines = rag_response.split('\n')
                for line in lines:
                    if 'SPECIALIST:' in line:
                        specialist = line.replace('SPECIALIST:', '').strip()
                        break
            
            # Get specific doctors from database
            doctor_list = ""
            if specialist:
                doctors = self.get_doctors_by_specialty(specialist)
                if doctors:
                    doctor_list = f"\n\n📋 **Available {specialist} Doctors:**\n"
                    for i, doctor in enumerate(doctors, 1):
                        name, spec, phone, location, experience = doctor
                        doctor_list += f"{i}. **{name}** ({spec})\n"
                        doctor_list += f"   📞 {phone} | 📍 {location} | ⭐ {experience} years\n\n"
            
            # Combine response
            final_response = rag_response + doctor_list
            final_response += "\n\n⚠️ **Please consult with a qualified doctor for proper diagnosis.**"
            
            return final_response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def refresh_database(self):
        """Refresh vector store with latest database data"""
        try:
            # Remove old vector store
            if os.path.exists("medical_vectorstore.pkl"):
                os.remove("medical_vectorstore.pkl")
            
            # Recreate from database
            documents = self.create_doctor_documents()
            if documents:
                self.vector_store = FAISS.from_documents(documents, embeddings)
                
                # Save new vector store
                with open("medical_vectorstore.pkl", "wb") as f:
                    pickle.dump(self.vector_store, f)
                
                # Recreate QA chain
                self.create_qa_chain()
                
                return "✅ Database refreshed successfully!"
            else:
                return "❌ No data found in database"
                
        except Exception as e:
            return f"❌ Error refreshing database: {str(e)}"

# Initialize bot
medical_bot = SimpleMedicalBot()

# Gradio Interface
def chat_function(message, history):
    return medical_bot.process_query(message, history)

def refresh_function():
    return medical_bot.refresh_database()

# Create interface
with gr.Blocks(theme=gr.themes.Soft(), title="Medical Bot") as demo:
    
    gr.Markdown("# 🏥 Medical Bot with Database RAG")
    gr.Markdown("### Get doctor recommendations based on your health problems")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.ChatInterface(
                fn=chat_function,
                examples=[
                    "I have fever and cough",
                    "Heart pain and chest discomfort", 
                    "Kidney problems",
                    "Skin rash and itching",
                    "Stomach pain and digestion issues",
                    "Bone and joint pain",
                    "Women's health issues",
                    "Eye problems and vision"
                ],
                title="Ask about your health problem",
                description="Describe your symptoms to get specialist recommendations"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Database Controls")
            
            refresh_btn = gr.Button("🔄 Refresh Database", variant="primary", size="lg")
            refresh_status = gr.Textbox(label="Status", interactive=False, lines=2)
            
            refresh_btn.click(
                fn=refresh_function,
                outputs=refresh_status
            )
            
            gr.Markdown("""
            ### 📊 Features:
            - **Real-time Database**: PostgreSQL integration
            - **Vector Search**: AI-powered doctor matching  
            - **Specialist Mapping**: Symptoms to specialists
            - **Location-based**: Find doctors by area
            - **Experience Ranking**: Sort by experience
            
            ### 🚀 How it works:
            1. Describe your health problem
            2. AI analyzes symptoms using database
            3. Get specialist recommendations
            4. See available doctors with contact info
            """)

if __name__ == "__main__":
    demo.launch(share=True)