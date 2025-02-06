import os
import re
import nltk
import streamlit as st
import PyPDF2
from langchain_community.document_loaders import TextLoader
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from groq import Groq
from dotenv import load_dotenv

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Fetch the API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=groq_api_key)

# Token limit for Groq API
TOKEN_LIMIT = 5000  

def preprocess_text(page_content):
    """Preprocess text (cleaning + tokenization)"""
    page_content = re.sub(r"\s+", " ", page_content)
    page_content = re.sub(r'[^\w\s.]', '', page_content)

    sentences = sent_tokenize(page_content)

    stop_words = set(stopwords.words('english'))
    cleaned_sentences = [
        ' '.join([word for word in sentence.split() if word.lower() not in stop_words])
        for sentence in sentences
    ]
    
    cleaned_text = ' '.join(cleaned_sentences)
    
    return cleaned_text  # Do NOT trim text here

def trim_text_to_limit(text, max_tokens=TOKEN_LIMIT):
    """Trim text if it exceeds the max token limit"""
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])  # Trim excess words
    return text

def generate_mcqs(text, num_questions=5):
    """Generate MCQs using Groq's API"""
    
    # Trim text only when sending it to Groq
    trimmed_text = trim_text_to_limit(text)

    prompt = f"""
    Generate {num_questions} multiple-choice questions (MCQs) based on the following text:
    {trimmed_text}
    Format:
    - Question: [Question]
      Options: A) [Option 1] 
               B) [Option 2] 
               C) [Option 3]
               D) [Option 4]
      
      **Answer**: [Correct Option]
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a helpful assistant that generates MCQs."},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating MCQs: {e}"

# Streamlit UI
st.title("MCQ Generator from Text or PDF File")
st.write("Upload a text or PDF file and specify the number of MCQs to generate (uploaded files shouldn't contain more than 6000 tokens as it's the limit for the model).")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=20, value=5)

file_content = ""

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
            st.write("✅ Successfully loaded text file.")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    file_content += text + "\n"  
            st.write("✅ Successfully loaded PDF file.")

    except UnicodeDecodeError:
        st.error("❌ Error: Invalid file encoding. Please upload a UTF-8 encoded text file.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

if file_content:
    # Preprocess but NOT trimming yet
    processed_text = preprocess_text(file_content)

    # Show original token count
    token_count = len(processed_text.split())
    st.write(f"📌 Processed text contains {token_count} tokens.")

    if st.button("Generate MCQs"):
        mcq_text = generate_mcqs(processed_text, num_questions)
        st.write("🎯 MCQs Generated:")
        st.write(mcq_text)
