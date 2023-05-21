import os
import logging
import wandb
import torch
from typing import Callable, Dict, Type
from datetime import datetime
from tqdm.autonotebook import tqdm, trange
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from src.bm25 import *
from load_dataset import load_corpus, load_queries, load_train, load_eval


class CustomCrossEncoder(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = {},
        automodel_args: Dict = {},
        default_activation_function=None,
    ):
        super().__init__(
            model_name,
            num_labels,
            max_length,
            device,
            tokenizer_args,
            automodel_args,
            default_activation_function,
        )

    def smart_batching_collate(self, batch):
        queries = []
        bm25 = []
        passages = []
        labels = []

        for example in batch:
            queries.append(example.texts[0])
            bm25.append(example.texts[1])
            passages.append(example.texts[2])
            labels.append(example.label)

        # Tokenize separately to control max_length for each
        tokenized_queries = self.tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt", max_length=30
        )
        tokenized_scores = self.tokenizer(
            bm25, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_passages = self.tokenizer(
            passages, padding=True, truncation=True, return_tensors="pt", max_length=200
        )

        # Concatenate query, bm, passage tokens along the sequence length dimension
        tokenized = {
            "input_ids": torch.cat(
                [
                    tokenized_queries["input_ids"],
                    tokenized_scores["input_ids"],
                    tokenized_passages["input_ids"],
                ],
                dim=-1,
            ),
            "attention_mask": torch.cat(
                [
                    tokenized_queries["attention_mask"],
                    tokenized_scores["attention_mask"],
                    tokenized_passages["attention_mask"],
                ],
                dim=-1,
            ),
        }
        labels = torch.tensor(
            labels, dtype=torch.float if self.config.num_labels == 1 else torch.long
        ).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        patience: int = None,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ):
        wandb.init(
            # set the wandb project where this run will be logged
            project="Information Retrieval Project",
            config={
                "epochs": epochs,
                "patience": patience,
                "warmup_steps": warmup_steps,
                "evaluation_steps": evaluation_steps,
                "optimizer_params": optimizer_params,
                "optimizer_class": optimizer_class.__name__,
            },
        )
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.counter = 0
        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

        if loss_fct is None:
            loss_fct = (
                nn.BCEWithLogitsLoss()
                if self.config.num_labels == 1
                else nn.CrossEntropyLoss()
            )

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                    wandb.log({"loss": loss_value})
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    wandb.log({"loss": loss_value})
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if (
                    evaluator is not None
                    and evaluation_steps > 0
                    and training_steps % evaluation_steps == 0
                ):
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps,
                        callback,
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                try:
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        -1,
                        callback,
                        patience,
                    )
                except EarlyStopping:
                    break

    def _eval_during_training(
        self,
        evaluator,
        output_path,
        save_best_model,
        epoch,
        steps,
        callback,
        patience=None,
    ):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
            else:
                self.counter += 1
                if patience is not None and self.counter >= patience:
                    raise EarlyStopping


class EarlyStopping(Exception):
    pass


def eval_callback(score, epoch, step):
    wandb.log({"MRR": score})


if __name__ == "__main__":
    wandb.login()
    # First, we define the transformer model we want to fine-tune
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    train_batch_size = 32
    num_epochs = 10
    model_save_path = (
        "output/training_ms-marco_cross-encoder-"
        + model_name.replace("/", "-")
        + "-"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # We set num_labels=1, which predicts a continous score between 0 and 1
    model = CustomCrossEncoder(model_name, num_labels=1, max_length=512)

    # Now we read the MS Marco dataset
    data_folder = "./data/msmarco-passage/"
    corpus = load_corpus(os.path.join(data_folder, "collection.tsv"))
    queries = load_queries(os.path.join(data_folder, "queries.train.tsv"))
    dev_samples = load_eval(
        os.path.join(data_folder, "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz"),
        corpus,
        queries,
    )
    train_samples = load_train(
        os.path.join(data_folder, "msmarco.bm25.train.tsv.gz"), corpus, queries
    )
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size
    )

    # We add an evaluator, which evaluates the performance during training
    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

    # Configure the training
    warmup_steps = 5000
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        patience=4,
        # loss_fct=nn.CrossEntropyLoss(),
        evaluation_steps=10_000,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_amp=True,
        # optimizer_class=Adam,
        # optimizer_params={'lr': 7e-6},
        callback=eval_callback,
    )

    # Save latest model
    model.save(model_save_path + "-latest")
