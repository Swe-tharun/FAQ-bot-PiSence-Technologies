import pandas as pd
from datasets import Dataset
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("data/faq.csv")

# Ensure required columns exist
if not all(col in df.columns for col in ["Question", "Context", "Answer"]):
    raise ValueError("CSV must contain 'Question', 'Context', and 'Answer' columns.")

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train and eval (80% train, 20% eval)
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Function to tokenize and compute answer positions
def preprocess_function(examples):
    inputs = tokenizer(
        examples["Question"], 
        examples["Context"], 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_offsets_mapping=True
    )
    
    start_positions = []
    end_positions = []
    
    for i in range(len(examples["Answer"])):
        answer = str(examples["Answer"][i])  # Ensure answer is a string
        context = str(examples["Context"][i])  # Ensure context is a string

        start_char = context.find(answer)
        if start_char == -1:
            print(f"Warning: Answer '{answer}' not found in Context!")
            start_positions.append(0)
            end_positions.append(0)
            continue  # Skip this sample if answer is not found in context

        end_char = start_char + len(answer) - 1
        offset_mapping = inputs["offset_mapping"][i]
        
        # Convert character positions to token positions
        token_start = None
        token_end = None

        for j, (start, end) in enumerate(offset_mapping):
            if start_char >= start and start_char < end:
                token_start = j
            if end_char >= start and end_char < end:
                token_end = j
                break

        if token_start is None or token_end is None:
            token_start = 0
            token_end = 0

        start_positions.append(token_start)
        end_positions.append(token_end)

    inputs.pop("offset_mapping")  # Remove unused offset mapping
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load model
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models",  
    evaluation_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    save_strategy="epoch",  
    save_total_limit=2,  
    logging_dir="./logs",  
    logging_steps=10,  
    report_to="none",  
    push_to_hub=False,  
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  
    eval_dataset=tokenized_dataset["test"],  
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./models/fqa-distilbert")
tokenizer.save_pretrained("./models/fqa-distilbert")

print("Training complete! Model saved to './models/fqa-distilbert'")
