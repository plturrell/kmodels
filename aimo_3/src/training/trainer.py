"""Training pipeline for fine-tuning LLMs on AIMO problems."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from .metrics import AIMOMetrics, compute_penalized_accuracy


class AIMODataset(Dataset):
    """Dataset for AIMO problems."""

    def __init__(self, problems: List[Dict], tokenizer, max_length: int = 2048):
        """
        Initialize dataset.

        Args:
            problems: List of problem dictionaries with 'statement' and 'answer'
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        statement = problem["statement"]
        answer = problem.get("answer", 0)

        # Create prompt
        prompt = f"Problem: {statement}\nAnswer: {answer}"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
        }


class AIMOTrainer:
    """Trainer for fine-tuning LLMs on AIMO problems."""

    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.

        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save checkpoints
            config: Training configuration
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.metrics = AIMOMetrics()

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        train_problems: List[Dict],
        val_problems: Optional[List[Dict]] = None,
    ):
        """
        Train the model.

        Args:
            train_problems: Training problems
            val_problems: Validation problems (optional)
        """
        if self.model is None:
            self.load_model()

        # Create datasets
        train_dataset = AIMODataset(
            train_problems,
            self.tokenizer,
            max_length=self.config.get("max_seq_length", 2048),
        )

        val_dataset = None
        if val_problems:
            val_dataset = AIMODataset(
                val_problems,
                self.tokenizer,
                max_length=self.config.get("max_seq_length", 2048),
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            learning_rate=self.config.get("learning_rate", 1e-5),
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=100 if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            fp16=True,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Evaluate on validation set
        if val_dataset:
            print("Evaluating on validation set...")
            eval_results = trainer.evaluate()
            print(f"Validation metrics: {eval_results}")

            # Compute custom AIMO metrics
            val_metrics = self._compute_aimo_metrics(val_problems)
            print(f"AIMO metrics: {val_metrics}")

            # Save metrics
            metrics_path = self.output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "transformers_metrics": eval_results,
                    "aimo_metrics": val_metrics,
                }, f, indent=2)

        # Save final model
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        print(f"Training complete. Model saved to {self.output_dir}")

    def _compute_aimo_metrics(self, problems: List[Dict]) -> Dict[str, float]:
        """Compute AIMO-specific metrics on problems."""
        if not problems or self.model is None:
            return {}

        self.metrics.reset()

        # Generate predictions
        predictions = []
        targets = []
        problem_ids = []

        for problem in problems:
            statement = problem["statement"]
            target = problem.get("answer", 0)
            problem_id = problem.get("problem_id", "unknown")

            # Generate prediction (simplified - would need proper inference)
            # This is a placeholder - actual implementation would use the model
            predictions.append(0)  # Placeholder
            targets.append(target)
            problem_ids.append(problem_id)

        self.metrics.update(predictions, targets, problem_ids)
        return self.metrics.compute()

    def save_config(self):
        """Save training configuration."""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

