# %%
from torch.optim import Optimizer
from torch import nn
from sentence_transformers.evaluation import SentenceEvaluator
import torch
from typing import Callable, Dict, Type
from tqdm.autonotebook import tqdm, trange
import tarfile
import os
import gzip
from datetime import datetime
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import LoggingHandler, util
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
!pip - q install wandb


# %%
wandb.login()


# %%


# %%


class CustomCrossEncoder(CrossEncoder):
    def __init__(self, model_name: str, num_labels: int = None, max_length: int = None, device: str = None, tokenizer_args: Dict = {},
                 automodel_args: Dict = {}, default_activation_function=None):
        super().__init__(model_name, num_labels, max_length, device,
                         tokenizer_args, automodel_args, default_activation_function)

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        wandb.init(
            # set the wandb project where this run will be logged
            project="Information Retrieval Project",
            config={
                "epochs": epochs,
                "warmup_steps": warmup_steps,
                "evaluation_steps": evaluation_steps,
                "optimizer_params": optimizer_params,
                "optimizer_class": optimizer_class.__name__,
            }
        )
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss(
            ) if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(
                            **features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                    wandb.log({"loss": loss_value})
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(
                        **features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    wandb.log({"loss": loss_value})
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(
                    evaluator, output_path, save_best_model, epoch, -1, callback)


def eval_callback(score, epoch, step):
    wandb.log({'MRR': score})


# %%

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout


# First, we define the transformer model we want to fine-tune
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 32
num_epochs = 1
model_save_path = 'output/training_ms-marco_cross-encoder-' + \
    model_name.replace("/", "-")+'-' + \
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

# Maximal number of training samples we want to use
max_train_samples = 2e7
# max_train_samples = 8_800_000


# We set num_labels=1, which predicts a continous score between 0 and 1
model = CustomCrossEncoder(model_name, num_labels=1, max_length=512)


# Now we read the MS Marco dataset
data_folder = './datasets/collectionandqueries'
os.makedirs(data_folder, exist_ok=True)


# %%
# Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get(
            'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


# Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get(
            'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query


# Now we create our training & dev data
train_samples = []
dev_samples = {}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200

# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
train_eval_filepath = os.path.join(
    data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download "+os.path.basename(train_eval_filepath))
    util.http_get(
        'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': queries[qid],
                                'positive': set(), 'negative': set()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].add(corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(corpus[neg_id])


# Read our training file
train_filepath = os.path.join(
    data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download "+os.path.basename(train_filepath))
    util.http_get(
        'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

cnt = 0
with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split()

        if qid in dev_samples:
            continue

        query = queries[qid]
        if (cnt % (pos_neg_ration+1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(
    train_samples, shuffle=True, batch_size=train_batch_size)


# %%
# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          loss_fct=nn.CrossEntropyLoss(),
          evaluation_steps=100,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True,
          optimizer_class=Adam,
          optimizer_params={'lr': 7e-6},
          callback=eval_callback)

# Save latest model
model.save(model_save_path+'-latest')


# %%
