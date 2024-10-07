import nltk
import os
import PyPDF2
from collections import Counter
import pandas as pd
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
import torch
import ollama
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data for text processing
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

def process_pdfs_in_folder(folder_path):
    """
    Process all PDF files in a given folder and extract their text content.
    
    Args:
    folder_path (str): Path to the folder containing PDF files.
    
    Returns:
    str: Concatenated text from all PDFs.
    """
    total_text = []

    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing: {pdf_path}")

        # Use PyPDF2 to extract text from PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
        
        total_text.append(text)

    # Combine text from all PDFs
    return "\n\n".join(total_text)

def nltk_based_splitter(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split the input text into chunks using NLTK's sentence tokenizer.
    
    Args:
    text (str): Input text to be split.
    chunk_size (int): Maximum size of each chunk.
    overlap (int): Number of characters to overlap between chunks.
    
    Returns:
    list: List of text chunks.
    """
   

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    # Create chunks of sentences
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Create overlapping chunks if overlap > 0
    if overlap > 0:
        overlapping_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                start_overlap = max(0, len(chunks[i-1]) - overlap)
                chunk_with_overlap = chunks[i-1][start_overlap:] + " " + chunks[i]
                overlapping_chunks.append(chunk_with_overlap[:chunk_size])
            else:
                overlapping_chunks.append(chunks[i][:chunk_size])

        return overlapping_chunks

    return chunks

def generate_with_ollama(text_chunk: str, temperature: float, model_name: str):
    """
    Generate a question-answer pair from a text chunk using Ollama.
    
    Args:
    text_chunk (str): Input text chunk.
    temperature (float): Temperature for text generation.
    model_name (str): Name of the Ollama model to use.
    
    Returns:
    tuple: Generated question and answer.
    """
    prompt = f"""
    Based on the following text, generate one Question and its corresponding Answer.
    Please format the output as follows:
    Question: [Your question]
    Answer: [Your answer]

    Text: {text_chunk}
    """
    
    # Generate response using Ollama
    response = ollama.generate(model=model_name, 
                               prompt=prompt)
    
    time.sleep(5)  # Pause to avoid rate limiting, adjust as needed

    # Extract question and answer from the response
    try:
        question, answer = response['response'].split("Answer:", 1)
        question = question.replace("Question:", "").strip()
        answer = answer.strip()
    except ValueError:
        question, answer = "N/A", "N/A"

    return question, answer

def process_text_chunks(text_chunks: list, temperature: float, model_name: str):
    """
    Process a list of text chunks to generate question-answer pairs.
    
    Args:
    text_chunks (list): List of text chunks.
    temperature (float): Temperature for text generation.
    model_name (str): Name of the Ollama model to use.
    
    Returns:
    pandas.DataFrame: DataFrame containing text chunks, questions, and answers.
    """
    results = []

    for chunk in text_chunks:
        time.sleep(5) 
        question, answer = generate_with_ollama(chunk, temperature, model_name)
        results.append({"Text_Chunk": chunk, "Question": question, "Answer": answer})

    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    print("Starting PDF processing...")
    folder_path = "data"
    all_text = process_pdfs_in_folder(folder_path)
    print(f"Total text length: {len(all_text)} characters")

    print("Splitting text into chunks...")
    chunks = nltk_based_splitter(text=all_text, chunk_size=2048, overlap=0)
    print(f"Number of chunks: {len(chunks)}")

    print("Generating Q&A pairs...")
    df_qa_pairs = process_text_chunks(text_chunks=chunks,
                                      temperature=0.7,
                                      model_name="phi3")
    print(f"Generated {len(df_qa_pairs)} Q&A pairs")

    print("Saving Q&A pairs to CSV...")
    df_qa_pairs.to_csv("ollama_generated_qa_pairs.csv", index=False)
    print("CSV file saved successfully")

    # Load the dataset from the CSV file
    dataset = load_dataset('csv', data_files='ollama_generated_qa_pairs.csv')

    def process_example(example, idx):
        """Process each example in the dataset."""
        return {
            "id": idx,
            "anchor": example["Question"],
            "positive": example["Answer"]
        }

    # Map the processing function to the dataset
    dataset = dataset.map(process_example,
                          with_indices=True,
                          remove_columns=["Text_Chunk", "Question", "Answer"])

    # Load the pre-trained model
    model_id = "BAAI/bge-base-en-v1.5"
    model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

    # Set up training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="bge-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
    )

    # Prepare corpus and queries for evaluation
    corpus = dict(zip(dataset['train']['id'], dataset['train']['positive']))
    queries = dict(zip(dataset['train']['id'], dataset['train']['anchor']))
    relevant_docs = {q_id: [q_id] for q_id in queries}

    # Create evaluators for different embedding dimensions
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    evaluator = SequentialEvaluator(matryoshka_evaluators)

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset.select_columns(["positive", "anchor"]),
        loss=train_loss,
        evaluator=evaluator,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model()

    # Evaluate the fine-tuned model
    fine_tuned_model = SentenceTransformer(args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    results = evaluator(fine_tuned_model)

    # Print evaluation results
    for dim in matryoshka_dimensions:
        key = f"dim_{dim}_cosine"
        print(f"{key}: {results[key]}")