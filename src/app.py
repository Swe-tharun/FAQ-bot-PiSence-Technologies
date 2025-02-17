import torch
import streamlit as st
import pandas as pd
import time
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---- Set Page Configuration (MUST be the first Streamlit command) ----
st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🎤",
    layout="centered",
)

# ---- Set Background Image Function ----
def set_background():
    st.markdown("""
        <style>
            .stApp {
                background: url("https://cdn.wallpapersafari.com/7/90/BFUQb1.jpg");
                background-size: cover;
                background-position: center;
            }
        </style>
    """, unsafe_allow_html=True)

# Call this function to set the background
set_background()

# ---- Custom CSS for Styling ----
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom,rgb(255, 255, 255), #2a0050);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .question-box {
            background: rgb(248, 246, 246);
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: black;
        }
        .answer-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            margin-top: 10px;
            color: white;
        }
        .wave-container {
            width: 100%;
            text-align: center;
            margin-top: 20px;
        }
        .wave {
            display: inline-block;
            width: 5px;
            height: 20px;
            background: #8A2BE2;
            margin: 0 2px;
            animation: wave-animation 1.2s infinite ease-in-out;
        }
        .wave:nth-child(2) { animation-delay: -1.1s; }
        .wave:nth-child(3) { animation-delay: -1.0s; }
        .wave:nth-child(4) { animation-delay: -0.9s; }
        .wave:nth-child(5) { animation-delay: -0.8s; }
        @keyframes wave-animation {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(1.5); }
        }
        h1 {
            color: white;
        }
        p {
            color: white;
        }
        /* Button Styling */
        .stButton > button {
            background-color: red;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: darkred;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load Model & Tokenizer ----
model_path = "models/fqa-distilbert"
model = DistilBertForQuestionAnswering.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# ---- Load FAQ Dataset ----
df = pd.read_csv("data/faq.csv")

# ---- Load Sentence Transformer for Semantic Search ----
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
df["Embedding"] = df["Question"].apply(lambda x: embedder.encode(x, convert_to_tensor=True))

# ---- UI Layout ----
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/piscienselogo.png", width=80)  # Adjust width as needed
with col2:
    st.markdown("<h1 style='color: #ede8ed;'>AI FAQ Assistant</h1>", unsafe_allow_html=True)

st.write("💡 *Ask any question and get an instant response!*")

# ---- Chat History Storage ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- User Input ----
question = st.text_input("🔍 Type your question here:")

if question:
    with st.spinner("Processing..."):
        time.sleep(1)  # Simulate processing delay
        
        # Compute similarity
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        similarities = [cosine_similarity(question_embedding.unsqueeze(0), emb.unsqueeze(0)).item() for emb in df["Embedding"]]

        best_match_idx = similarities.index(max(similarities))
        best_context = df.iloc[best_match_idx]["Context"]

        # ---- Tokenize with question and context ----
        inputs = tokenizer(question, best_context, return_tensors="pt", truncation=True)

        # ---- Get model predictions ----
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract answer start and end
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        # Convert tokens to string
        if answer_start >= answer_end:
            answer = "❌ Sorry, I couldn't find an answer."
        else:
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
            )

        # Save current question and answer to chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})

        # ---- Display answer ----
        st.markdown(f"<div class='question-box'><b>Q:</b> {question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-box'><b>A:</b> {answer}</div>", unsafe_allow_html=True)

# ---- Delete Specific History Functionality ----
def delete_chat(index):
    st.session_state.chat_history.pop(index)

# ---- Chat History Button and Toggle Visibility ----
if st.button("Chat History"):
    if st.session_state.chat_history:
        st.subheader("📜 Previous Questions")
        for idx, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='question-box'><b>Q:</b> {chat['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'><b>A:</b> {chat['answer']}</div>", unsafe_allow_html=True)
            # Add a delete button for each chat
            if st.button(f"Delete Chat {idx + 1}", key=f"delete_{idx}"):
                delete_chat(idx)
                st.experimental_rerun()  # Refresh to reflect changes
    else:
        st.write("No chat history yet.")

# ---- Animated Voice Wave Effect ----
st.markdown("""
    <div class="wave-container">
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
    </div>
""", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("<br><hr><p style='text-align:center;'>🔹 Powered by PiSence AI | © 2025 PiSence Technologies</p>", unsafe_allow_html=True)
