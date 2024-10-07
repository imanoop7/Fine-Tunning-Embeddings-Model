# Fine-Tuning Embeddings Model

This project focuses on fine-tuning a pre-trained embedding model (BAAI/bge-base-en-v1.5) using custom question-answer pairs generated from PDF documents.

## Project Overview

1. PDF Processing: Extract text from PDF files.
2. Text Chunking: Split extracted text into manageable chunks.
3. Q&A Generation: Generate question-answer pairs using the Groq API.
4. Dataset Preparation: Create a dataset from the generated Q&A pairs.
5. Model Fine-Tuning: Fine-tune the BAAI/bge-base-en-v1.5 model using the prepared dataset.
6. Evaluation: Assess the performance of the fine-tuned model.
7. Push to Hugging Face: Upload the fine-tuned model to Hugging Face Hub.

## Requirements

See `requirements.txt` for a full list of dependencies.

## Usage

1. Place your PDF files in the `data` folder.
2. Set up your Groq API key as an environment variable:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```
3. Run the main script:
   ```bash
   python embedding_main.py
   ```
4. To push the fine-tuned model to Hugging Face Hub, run:
   ```bash
   python push_to_huggingface.py
   ```

## Note on Training Arguments

Due to limited resources, we have used very simple training arguments in this project. You can increase these parameters according to your needs and available computational resources. The current settings are:

```python
args = SentenceTransformerTrainingArguments(
        output_dir="bge-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        fp16=False,  # Disable mixed precision training
        bf16=False,  # Disable bfloat16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
    )
```

## Pushing to Hugging Face

After fine-tuning the model, you can use the `push_to_huggingface.py` script to upload your model to the Hugging Face Hub. Make sure to set your Hugging Face API token as an environment variable:

```
export HF_TOKEN=your_huggingface_token_here
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
