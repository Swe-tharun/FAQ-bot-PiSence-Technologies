Here’s a **well-structured README** for your FAQ Chatbot project:  

---

## **📌 FAQ Chatbot using DistilBERT**
An AI-powered **FAQ Chatbot** that predicts answers from a dataset of frequently asked questions. The dataset was created by **scraping the PiSence Technologies website** and fine-tuning **DistilBERT** for question answering.

---

## **🚀 Project Overview**
This chatbot uses **Natural Language Processing (NLP)** to find relevant answers for user queries. It first finds the most similar question using **Sentence Transformers** and then extracts the precise answer using **DistilBERT**.

### **🔍 How It Works**
1. **Web Scraping** – Extracted FAQs from the **PiSence Technologies website**.
2. **Dataset Preparation** – Cleaned and structured the extracted data.
3. **Fine-Tuning** – Trained **DistilBERT** (`distilbert-base-uncased`) on the dataset.
4. **Embedding Similarity** – Used `paraphrase-MiniLM-L6-v2` for ranking relevant FAQs.
5. **Question Answering** – The chatbot extracts the most relevant answer.

---

## **📑 Technologies Used**
✅ **Python**  
✅ **Streamlit** (For UI)  
✅ **PyTorch** (For fine-tuning)  
✅ **Transformers** (Hugging Face library)  
✅ **Sentence Transformers** (For semantic similarity)  
✅ **Pandas** (For dataset handling)  
✅ **BeautifulSoup** (For web scraping)  

---

## **🎯 Model Performance**
- **Accuracy**: 87% on the test dataset  
- **Dataset**: Scraped from **PiSence Technologies**  
- **Fine-Tuned Model**: `distilbert-base-uncased`  

---

## **📂 Installation & Setup**
### **🔹 Clone the Repository**
```bash
git clone https://github.com/your-username/faq-chatbot.git
cd faq-chatbot
```

### **🔹 Install Dependencies**
```bash
pip install -r requirements.txt
```

### **🔹 Download & Setup the Model**
If the model is not found, run:
```bash
chmod +x download_models.sh
./download_models.sh
```

Or, you can manually download:
```bash
huggingface-cli download distilbert-base-uncased-distilled-squad --local-dir models/fqa-distilbert
```

### **🔹 Run the Chatbot**
```bash
streamlit run app.py
```

---

## **🛠 Troubleshooting**
### **1️⃣ Import Error: `Dilbert is not correctly imported`**
- **Solution**: Restart the local server:
```bash
streamlit run app.py
```

### **2️⃣ Model Not Found**
- **Solution**: Run the model download script again:
```bash
./download_models.sh
```

### **3️⃣ Streamlit Not Found**
- **Solution**: Install it:
```bash
pip install streamlit
```
## 🚀 Common Errors While Training

### 1️⃣ What if the model `fqa-distilbert` or `distilbert-base-uncased` is not found?  
✅ After running `train.py`, a folder named **models/fqa-distilbert** will be **automatically created** with the trained model.  
⚠️ If it's missing, ensure the training script has been executed successfully.

### 2️⃣ How can I improve the model accuracy?  
🔹 **Increase the batch size** (`per_device_train_batch_size=16` or higher, if GPU allows).  
🔹 **Increase the number of epochs** (e.g., `num_train_epochs=15` for better learning).  
🔹 **Use a larger model** like `bert-base-uncased` instead of `distilbert-base-uncased`.  

---

## **📌 Future Enhancements**
- ✅ Add **voice-based query support** 🎤  
- ✅ Improve **response accuracy** with more training data 📊  
- ✅ Deploy as a **web app** using **AWS/GCP** 🌐  

---

## **📞 Contact**
📧 **Email**: tharunsivaraj0325@gmail.com  
🔗 **GitHub**: https://github.com/Swe-tharun

🚀 **Happy Coding!** 😊
