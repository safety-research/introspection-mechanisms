"""
Utilities for supervised fine-tuning with LoRA.

This module handles:
- Dataset creation for various cognitive tasks
- LoRA fine-tuning pipeline
- Task performance evaluation
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_anomaly_detection_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """
    Create anomaly detection dataset with normal and anomalous sentences.

    Args:
        n: Total number of examples
        seed: Random seed

    Returns:
        HuggingFace Dataset with 'text' and 'label' columns
    """
    set_seed(seed)

    # Normal sentences (clean, grammatical)
    normal_sentences = [
        "The cat sat on the mat.",
        "She enjoys reading books in the library.",
        "The sun sets in the west.",
        "He plays guitar every evening.",
        "The flowers bloom in spring.",
        "They walked through the park together.",
        "Coffee tastes better in the morning.",
        "The train arrives at noon.",
        "Children play in the playground.",
        "The moon shines at night.",
        "Birds fly south for the winter.",
        "She painted a beautiful landscape.",
        "The river flows to the ocean.",
        "He solved the math problem quickly.",
        "The library closes at six o'clock.",
        "Dogs bark when strangers approach.",
        "Rain falls from the clouds.",
        "The teacher explains the lesson clearly.",
        "Mountains rise above the valley.",
        "She writes in her journal daily.",
    ]

    # Anomalous sentence patterns
    anomaly_templates = [
        # Semantic violations
        "The rock ate quickly.",
        "The color seven smells purple.",
        "The truth was very rectangular.",
        "He drank a solid wall.",
        "The silence was deafeningly visible.",
        # Factual errors
        "Two plus two equals seven.",
        "Paris is the capital of Germany.",
        "The Earth is flat.",
        "Water boils at 50 degrees Celsius.",
        "Humans have three arms.",
        # Grammatical errors
        "He go to store yesterday.",
        "She don't likes apples.",
        "They was playing soccer.",
        "The cat are sleeping.",
        "I has a book.",
        # Word substitutions (nonsensical)
        "The bicycle telephoned brightly.",
        "She elephant the tomorrow.",
        "The happiness walked under purple.",
        "He mathematics the beautiful yesterday.",
        "The clouds keyboard very softly.",
    ]

    examples = []

    # Generate examples
    for i in range(n):
        if i < n // 2:
            # Normal examples
            sentence = random.choice(normal_sentences)
            # Add some variation
            if random.random() < 0.3:
                sentence = sentence.replace(".", random.choice([".",".",".","!"]))
            label = "Normal"
        else:
            # Anomalous examples
            sentence = random.choice(anomaly_templates)
            label = "Unusual"

        # Format as Q&A
        text = f"Is this sentence normal or unusual? {sentence}\nAnswer: {label}"
        examples.append({"text": text, "label": label})

    return Dataset.from_list(examples)


def create_error_detection_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """
    Create error detection dataset using mathematical reasoning.

    Args:
        n: Total number of examples
        seed: Random seed

    Returns:
        HuggingFace Dataset
    """
    set_seed(seed)

    examples = []

    for i in range(n):
        if i < n // 2:
            # Correct examples
            a, b = random.randint(1, 50), random.randint(1, 50)
            operation = random.choice(['+', '-', '*'])

            if operation == '+':
                result = a + b
                solution = f"Adding {a} and {b} gives us {result}."
            elif operation == '-':
                result = a - b
                solution = f"Subtracting {b} from {a} gives us {result}."
            else:
                result = a * b
                solution = f"Multiplying {a} by {b} gives us {result}."

            question = f"What is {a} {operation} {b}?"
            label = "No"
        else:
            # Incorrect examples
            a, b = random.randint(1, 50), random.randint(1, 50)
            operation = random.choice(['+', '-', '*'])

            if operation == '+':
                correct = a + b
                result = correct + random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                solution = f"Adding {a} and {b} gives us {result}."
            elif operation == '-':
                correct = a - b
                result = correct + random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                solution = f"Subtracting {b} from {a} gives us {result}."
            else:
                correct = a * b
                result = correct + random.choice([-10, -5, -2, 2, 5, 10])
                solution = f"Multiplying {a} by {b} gives us {result}."

            question = f"What is {a} {operation} {b}?"
            label = "Yes"

        text = f"Does this solution contain an error?\nQuestion: {question}\nSolution: {solution}\nAnswer: {label}"
        examples.append({"text": text, "label": label})

    return Dataset.from_list(examples)


def create_metacognitive_calibration_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """
    Create metacognitive calibration dataset with confidence judgments.

    Uses trivia questions with correct/incorrect answers and confidence scores.
    """
    set_seed(seed)

    # Simple trivia questions
    trivia = [
        ("What is the capital of France?", "Paris", 95),
        ("What is 2+2?", "4", 100),
        ("What year did World War II end?", "1945", 85),
        ("What is the largest planet in our solar system?", "Jupiter", 90),
        ("Who wrote Romeo and Juliet?", "William Shakespeare", 95),
        ("What is the speed of light?", "299,792,458 m/s", 70),
        ("What is the chemical symbol for gold?", "Au", 85),
        ("How many continents are there?", "7", 90),
        ("What is the smallest prime number?", "2", 95),
        ("What is the boiling point of water?", "100°C", 95),
    ]

    examples = []
    for i in range(n):
        q, correct_ans, base_conf = random.choice(trivia)

        # Add noise to confidence
        confidence = max(0, min(100, base_conf + random.randint(-10, 10)))

        # Sometimes use incorrect answer with lower confidence
        if random.random() < 0.3:
            answer = "I don't know"
            confidence = random.randint(0, 40)
        else:
            answer = correct_ans

        text = f"Question: {q}\nAnswer: {answer}\nConfidence (0-100): {confidence}"
        examples.append({"text": text, "confidence": confidence})

    return Dataset.from_list(examples)


def create_self_consistency_dataset(n: int = 3000, seed: int = 42) -> Dataset:
    """
    Create dataset for detecting inconsistent statement pairs.

    Args:
        n: Number of statement pairs
        seed: Random seed

    Returns:
        HuggingFace Dataset
    """
    set_seed(seed)

    # Consistent pairs
    consistent_pairs = [
        ("Paris is the capital of France.", "The Eiffel Tower is in Paris."),
        ("Water freezes at 0°C.", "Ice is frozen water."),
        ("The Earth orbits the Sun.", "It takes one year for Earth to complete its orbit."),
        ("Dogs are mammals.", "Mammals are warm-blooded animals."),
        ("Python is a programming language.", "You can write code in Python."),
    ]

    # Inconsistent pairs
    inconsistent_pairs = [
        ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ("Water freezes at 0°C.", "Water freezes at 50°C."),
        ("The Earth is round.", "The Earth is flat."),
        ("Today is Monday.", "Today is Friday."),
        ("2+2=4", "2+2=7"),
    ]

    examples = []
    for i in range(n):
        if i < n // 2:
            s1, s2 = random.choice(consistent_pairs)
            label = "Yes"
        else:
            s1, s2 = random.choice(inconsistent_pairs)
            label = "No"

        text = f"Are these statements consistent?\nStatement 1: {s1}\nStatement 2: {s2}\nAnswer: {label}"
        examples.append({"text": text, "label": label})

    return Dataset.from_list(examples)


def create_cot_reasoning_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """
    Create chain-of-thought reasoning dataset using GSM8K-style problems.
    """
    set_seed(seed)

    examples = []

    for i in range(n):
        # Generate simple word problems
        item_price = random.randint(2, 20)
        quantity = random.randint(2, 10)
        total = item_price * quantity

        problem = f"If each apple costs ${item_price} and you buy {quantity} apples, how much do you pay?"
        cot = f"Let's think step by step. Each apple costs ${item_price}. We're buying {quantity} apples. So the total cost is {quantity} × ${item_price} = ${total}."

        text = f"Problem: {problem}\nSolution: {cot}"
        examples.append({"text": text})

    return Dataset.from_list(examples)


def create_instruction_following_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """
    Create instruction following dataset.
    Uses simple task instructions and responses.
    """
    set_seed(seed)

    # Load from Alpaca-style dataset or create simple instructions
    instructions = [
        ("Write a greeting.", "Hello! How can I help you today?"),
        ("Count from 1 to 5.", "1, 2, 3, 4, 5"),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Translate 'hello' to Spanish.", "'Hello' in Spanish is 'hola'."),
        ("Name three colors.", "Three colors are red, blue, and green."),
    ]

    examples = []
    for i in range(n):
        instruction, response = random.choice(instructions)
        text = f"Instruction: {instruction}\nResponse: {response}"
        examples.append({"text": text})

    return Dataset.from_list(examples)


def load_squad_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """Load and format SQuAD 2.0 for QA training."""
    set_seed(seed)

    dataset = load_dataset("rajpurkar/squad_v2", split="train")

    # Sample and format
    dataset = dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))

    def format_squad(example):
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"]

        if len(answers) > 0:
            answer = answers[0]
        else:
            answer = "No answer found."

        text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        return {"text": text}

    return dataset.map(format_squad, remove_columns=dataset.column_names)


def load_sentiment_dataset(n: int = 5000, seed: int = 42) -> Dataset:
    """Load and format SST-2 for sentiment classification."""
    set_seed(seed)

    dataset = load_dataset("stanfordnlp/sst2", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))

    def format_sentiment(example):
        sentence = example["sentence"]
        label = "Positive" if example["label"] == 1 else "Negative"
        text = f"Classify the sentiment: {sentence}\nSentiment: {label}"
        return {"text": text}

    return dataset.map(format_sentiment, remove_columns=dataset.column_names)


def prepare_task_dataset(task_name: str, n_train: int = 5000, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Prepare train/val split for a specific task.

    Args:
        task_name: Name of the task
        n_train: Number of training examples
        seed: Random seed

    Returns:
        (train_dataset, val_dataset)
    """
    task_creators = {
        "anomaly_detection": create_anomaly_detection_dataset,
        "error_detection": create_error_detection_dataset,
        "metacognitive_calibration": create_metacognitive_calibration_dataset,
        "self_consistency": create_self_consistency_dataset,
        "cot_reasoning": create_cot_reasoning_dataset,
        "instruction_following": create_instruction_following_dataset,
        "standard_qa": load_squad_dataset,
        "sentiment_classification": load_sentiment_dataset,
    }

    if task_name not in task_creators:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_creators.keys())}")

    # Create full dataset
    total_n = int(n_train / 0.8)  # 80% train, 20% val
    full_dataset = task_creators[task_name](n=total_n, seed=seed)

    # Split
    split = full_dataset.train_test_split(test_size=0.2, seed=seed)

    return split["train"], split["test"]


def train_with_lora(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> None:
    """
    Fine-tune model with LoRA.

    Args:
        model: Base model
        tokenizer: Tokenizer
        train_dataset: Training dataset with 'text' column
        val_dataset: Validation dataset
        output_dir: Where to save checkpoints
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        device: Device to train on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model for k-bit training if quantized
    if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize datasets
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train set",
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val set",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Check if bf16 is supported
    use_bf16 = False
    if device == "cuda" and torch.cuda.is_available():
        # Check if GPU supports bfloat16
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            print("BF16 not supported, using FP32")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",  # Match eval_strategy
        save_steps=200,         # Match eval_steps
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Train
    print(f"Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / "final"))

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Training complete. Model saved to {output_dir / 'final'}")


def evaluate_task_performance(
    model,
    tokenizer,
    test_dataset: Dataset,
    task_name: str,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model performance on task test set.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        task_name: Task name
        max_length: Max sequence length
        device: Device

    Returns:
        Dict with metrics (loss, perplexity, etc.)
    """
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in tqdm(test_dataset, desc=f"Evaluating {task_name}"):
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "test_loss": avg_loss,
        "test_perplexity": perplexity,
    }
