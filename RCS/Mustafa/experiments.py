#running the 3 main experiments 
#exp 1 : training on emotion and polarity  , testing on emotion
#exp 2 : training on polarity and testing on polarity(baseline)
#exp 3 : training on both polarity and emotion and testing on polarity
import os
import json
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gc
import traceback
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
    DebertaTokenizer,
    EarlyStoppingCallback,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

def clear_memory_aggressive():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# REPRODUCIBILITY, didnt try to reproduce the results on another computer though...

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(0.95)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# CONFIG

TRAIN_FILE = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_train.jsonl"
VALID_FILE = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_validation.jsonl"
TEST_FILE  = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_test.jsonl"

MODELS = {
    "bert": {
        "name": "bert-base-uncased", "type": "transformer", "params": 110, "family": "BERT",
        "batch_size": 16, "grad_accum": 2, "epochs": 10, "patience": 3
    },
    "bert-large": {
        "name": "bert-large-uncased", "type": "transformer", "params": 340, "family": "BERT",
        "batch_size": 8, "grad_accum": 4, "epochs": 8, "patience": 3
    },
    "distilbert": {
        "name": "distilbert-base-uncased", "type": "transformer", "params": 66, "family": "DistilBERT",
        "batch_size": 32, "grad_accum": 1, "epochs": 12, "patience": 4
    },
    "roberta": {
        "name": "roberta-base", "type": "transformer", "params": 125, "family": "RoBERTa",
        "batch_size": 16, "grad_accum": 2, "epochs": 10, "patience": 3
    },
    "roberta-large": {
        "name": "roberta-large", "type": "transformer", "params": 355, "family": "RoBERTa",
        "batch_size": 8, "grad_accum": 4, "epochs": 8, "patience": 3
    },
    "distilroberta": {
        "name": "distilbert/distilroberta-base", "type": "transformer", "params": 82, "family": "DistilRoBERTa",
        "batch_size": 32, "grad_accum": 1, "epochs": 12, "patience": 4
    },
    "deberta": {
        "name": "microsoft/deberta-base", "type": "transformer", "params": 140, "family": "DeBERTa",
        "batch_size": 16, "grad_accum": 2, "epochs": 10, "patience": 3
    },
    "logistic_regression": {
        "name": "logistic_regression", "type": "traditional_ml", "params": 0, "family": "Traditional ML",
        "batch_size": None, "grad_accum": None, "epochs": None, "patience": None
    },
    "svm": {
        "name": "svm", "type": "traditional_ml", "params": 0, "family": "Traditional ML",
        "batch_size": None, "grad_accum": None, "epochs": None, "patience": None
    }
}

OUTPUT_ROOT = "./all_models_comparison_multitask"
MAX_LENGTH = 128
LR = 2e-5
LAMBDA_EMO = 1.0
LAMBDA_POL = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MULTITASK MODEL COMPARISON (NO POLARITY FED AT TEST INPUT)")
print("=" * 80)
print(f"Seed: {SEED}")
print(f"Device: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)


# HELPERS

def make_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {l: i for i, l in enumerate(uniq)}

def metrics_basic(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "f1_weighted": float(f1_weighted), "f1_macro": float(f1_macro)}

def encode(tokenizer, texts, aspects, max_length=128):
    combined = [f"aspect: {a} text: {t}" for a, t in zip(aspects, texts)]
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

def print_report(title, y_true, y_pred, id2label):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    m = metrics_basic(y_true, y_pred)
    print(f"Accuracy: {m['acc']:.4f}")
    print(f"F1 Weighted: {m['f1_weighted']:.4f}")
    print(f"F1 Macro: {m['f1_macro']:.4f}")
    labels_sorted = sorted(id2label.keys())
    names = [id2label[i] for i in labels_sorted]
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels_sorted,
        target_names=names,
        zero_division=0,
        digits=4
    ))
    return m

def load_tokenizer(model_key: str, model_name: str):
    if "deberta" in model_key.lower():
        return DebertaTokenizer.from_pretrained(model_name)
    if "roberta" in model_key.lower():
        return RobertaTokenizerFast.from_pretrained(model_name)
    if "distilbert" in model_key.lower():
        return DistilBertTokenizerFast.from_pretrained(model_name)
    if "bert" in model_key.lower():
        return BertTokenizerFast.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


# DATA LOADING

def load_jsonl_aspect_level_emotion_polarity(path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    texts, aspects, emotions, polarities = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "input" not in item or "output" not in item:
                continue
            review = item["input"]
            outputs = item["output"]
            if not isinstance(outputs, list):
                continue

            for a in outputs:
                if not isinstance(a, dict):
                    continue

                aspect = str(a.get("aspect", "")).lower().strip()
                emotion = str(a.get("emotion", "")).lower().strip()
                polarity = str(a.get("polarity", "")).lower().strip()

                def norm(x: str) -> str:
                    if x in ["", "masked"]:
                        return ""
                    if x == "null":
                        return "neutral"
                    return x

                emotion = norm(emotion)
                polarity = norm(polarity)

                if emotion == "" or polarity == "":
                    continue

                texts.append(review)
                aspects.append(aspect)
                emotions.append(emotion)
                polarities.append(polarity)

    return texts, aspects, emotions, polarities

class EncDataset(torch.utils.data.Dataset):
    def __init__(self, enc, y_em=None, y_pol=None, aspects=None):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.y_em = y_em
        self.y_pol = y_pol
        self.aspects = aspects

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.y_em is not None:
            item["labels_em"] = torch.tensor(self.y_em[idx], dtype=torch.long)
        if self.y_pol is not None:
            item["labels_pol"] = torch.tensor(self.y_pol[idx], dtype=torch.long)
        # kept for inspection only; Trainer will drop it (remove_unused_columns=True)
        if self.aspects is not None:
            item["aspect_str"] = self.aspects[idx]
        return item

def collate(batch):
    out = {}
    for k in batch[0].keys():
        if k == "aspect_str":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# TRANSFORMER MODELS

class BaseMultiTaskModel(nn.Module):
    def __init__(self, model_name: str, n_em: int, n_pol: int, head_hidden: int = 128,
                 lambda_em: float = 1.0, lambda_pol: float = 1.0,
                 class_weights_em=None, class_weights_pol=None):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        config.use_cache = False
        config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.use_pooler = hasattr(self.backbone, "pooler") and self.backbone.pooler is not None

        # HARD fp32 guard for DeBERTa (prevents fp16 masked_fill overflow)
        self.force_fp32 = ("deberta" in model_name.lower())

        h = config.hidden_size
        torch.manual_seed(SEED)
        self.head_em = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_em),
        )
        self.head_pol = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_pol),
        )

        self.lambda_em = lambda_em
        self.lambda_pol = lambda_pol
        self.ce_em = nn.CrossEntropyLoss(weight=class_weights_em) if class_weights_em is not None else nn.CrossEntropyLoss()
        self.ce_pol = nn.CrossEntropyLoss(weight=class_weights_pol) if class_weights_pol is not None else nn.CrossEntropyLoss()

        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels_em=None, labels_pol=None, **kwargs):
        # Ensure proper types
        if attention_mask is not None and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        # HARD disable autocast for DeBERTa even if accelerate enables it
        if self.force_fp32 and torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hs = outputs.hidden_states
        cls = outputs.pooler_output if self.use_pooler else hs[-1][:, 0, :]

        # keep heads in fp32 too when force_fp32
        if self.force_fp32:
            cls = cls.float()

        logits_em = self.head_em(cls)
        logits_pol = self.head_pol(cls)

        if labels_em is not None and labels_pol is not None:
            loss_em = self.ce_em(logits_em, labels_em)
            loss_pol = self.ce_pol(logits_pol, labels_pol)
            loss = self.lambda_em * loss_em + self.lambda_pol * loss_pol
            return {"loss": loss, "logits_em": logits_em, "logits_pol": logits_pol}

        return {"logits_em": logits_em, "logits_pol": logits_pol}

class BasePolarityOnlyModel(nn.Module):
    def __init__(self, model_name: str, n_pol: int, head_hidden: int = 128, class_weights=None):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        config.use_cache = False
        config.output_hidden_states = False
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.use_pooler = hasattr(self.backbone, "pooler") and self.backbone.pooler is not None

        # HARD fp32 guard for DeBERTa
        self.force_fp32 = ("deberta" in model_name.lower())

        h = config.hidden_size
        torch.manual_seed(SEED)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_pol),
        )

        self.ce = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels_pol=None, **kwargs):
        if attention_mask is not None and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        if self.force_fp32 and torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs.pooler_output if self.use_pooler and getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]

        if self.force_fp32:
            cls = cls.float()

        logits_pol = self.head(cls)

        if labels_pol is not None:
            loss = self.ce(logits_pol, labels_pol)
            return {"loss": loss, "logits_pol": logits_pol}
        return {"logits_pol": logits_pol}


# TRADITIONAL ML

class TraditionalMLWithPolarity:
    def __init__(self, model_type="logistic_regression", random_state=42, max_iter=1000):
        self.scaler = StandardScaler()
        self.polarity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._pol_dim = None

        if model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, multi_class="ovr", n_jobs=-1)
        elif model_type == "svm":
            self.model = LinearSVC(random_state=random_state, max_iter=max_iter, dual=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y_em, y_pol):
        pol_feat = self.polarity_encoder.fit_transform(y_pol.reshape(-1, 1))
        self._pol_dim = pol_feat.shape[1]
        Xc = np.hstack([X, pol_feat])
        Xs = self.scaler.fit_transform(Xc)
        self.model.fit(Xs, y_em)
        return self

    def predict(self, X):
        pol_zeros = np.zeros((X.shape[0], self._pol_dim), dtype=np.float32)
        Xc = np.hstack([X, pol_zeros])
        Xs = self.scaler.transform(Xc)
        return self.model.predict(Xs)

class TraditionalMLPolarityOnly:
    def __init__(self, model_type="logistic_regression", random_state=42, max_iter=1000):
        self.scaler = StandardScaler()
        if model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, multi_class="ovr", n_jobs=-1)
        elif model_type == "svm":
            self.model = LinearSVC(random_state=random_state, max_iter=max_iter, dual=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y_pol):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y_pol)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


# FEATURE EXTRACTOR (for Traditional ML)

class MemoryEfficientFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = False
        config.use_cache = False
        self.model = AutoModel.from_pretrained(model_name, config=config).to(self.device)
        self.model.eval()

    def extract_features_batch(self, texts, aspects, max_length=128, batch_size=16):
        combined = [f"aspect: {a} text: {t}" for a, t in zip(aspects, texts)]
        feats = []
        for i in tqdm(range(0, len(combined), batch_size), desc="Extracting features"):
            batch = combined[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    f = out.pooler_output.cpu().numpy()
                else:
                    f = out.last_hidden_state[:, 0, :].cpu().numpy()
            feats.append(f)
            del enc, out, f
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.vstack(feats)


# TRAINING ARGS
#   NOTE: DeBERTa still has fp16 disabled here, hard forcing fp32

def make_args(out_dir: str, batch_size: int, grad_accum: int, epochs: int, learning_rate: float,
              use_fp16: bool):
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        fp16=use_fp16,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        seed=SEED,
        data_seed=SEED,
        optim="adamw_torch",
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_only_model=True,
    )


# TRAINERS

class DeterministicMultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_em = inputs.pop("labels_em")
        labels_pol = inputs.pop("labels_pol")
        outputs = model(**inputs, labels_em=labels_em, labels_pol=labels_pol)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        g = torch.Generator(); g.manual_seed(SEED)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=g,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        g = torch.Generator(); g.manual_seed(SEED)
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=g,
            worker_init_fn=seed_worker,
        )

class DeterministicPolarityOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_pol = inputs.pop("labels_pol")
        outputs = model(**inputs, labels_pol=labels_pol)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        g = torch.Generator(); g.manual_seed(SEED)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=g,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        g = torch.Generator(); g.manual_seed(SEED)
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            generator=g,
            worker_init_fn=seed_worker,
        )

# Experiment 1: train polarity and emotion , test emotion

def run_experiment_1(tr_t, tr_a, tr_e, tr_p, va_t, va_a, va_e, va_p, te_t, te_a, te_e, te_p,
                     emotion2id, pol2id, id2emotion, class_weights_em_tensor, class_weights_pol_tensor):
    print("\n" + "="*80)
    print("EXPERIMENT 1: TRAIN ON BOTH EMOTION & POLARITY, TEST ON EMOTION (NO POLARITY FED)")
    print("="*80)

    ytr_e = np.array([emotion2id[x] for x in tr_e], dtype=np.int32)
    yva_e = np.array([emotion2id[x] for x in va_e], dtype=np.int32)
    yte_e = np.array([emotion2id[x] for x in te_e], dtype=np.int32)

    ytr_p = np.array([pol2id[x] for x in tr_p], dtype=np.int32)
    yva_p = np.array([pol2id[x] for x in va_p], dtype=np.int32)
    ytr_p = ytr_p.astype(np.int32)
    yva_p = yva_p.astype(np.int32)

    all_results = {}
    order = ["logistic_regression", "svm", "distilbert", "bert", "roberta",
             "distilroberta", "deberta", "bert-large", "roberta-large"]

    for model_key in order:
        if model_key not in MODELS:
            continue
        info = MODELS[model_key]
        print(f"\n{'='*80}\nEXPERIMENT 1: {model_key.upper()} ({info['name']})\n{'='*80}")
        clear_memory_aggressive()

        try:
            if info["type"] == "traditional_ml":
                extractor = MemoryEfficientFeatureExtractor("distilbert-base-uncased")
                Xtr = extractor.extract_features_batch(tr_t, tr_a, MAX_LENGTH, batch_size=8)
                Xte = extractor.extract_features_batch(te_t, te_a, MAX_LENGTH, batch_size=8)

                ml = TraditionalMLWithPolarity(model_type=model_key, random_state=SEED, max_iter=1000)
                ml.fit(Xtr, ytr_e, ytr_p)

                # TEST: no polarity fed
                yhat = ml.predict(Xte)
                metrics = print_report(
                    f"{model_key.upper()} (Train uses polarity, Test uses NO polarity) - Emotion Test",
                    yte_e, yhat, id2emotion
                )
                all_results[model_key] = {"emotion_test_no_polarity_input": metrics}

                del extractor, Xtr, Xte, ml
                clear_memory_aggressive()
                continue

            tokenizer = load_tokenizer(model_key, info["name"])
            enc_tr = encode(tokenizer, tr_t, tr_a, MAX_LENGTH)
            enc_va = encode(tokenizer, va_t, va_a, MAX_LENGTH)
            enc_te = encode(tokenizer, te_t, te_a, MAX_LENGTH)

            ds_tr = EncDataset(enc_tr, y_em=ytr_e, y_pol=ytr_p, aspects=tr_a)
            ds_va = EncDataset(enc_va, y_em=yva_e, y_pol=yva_p, aspects=va_a)

            # TEST: inputs only (no labels)
            ds_te_inputs = EncDataset(enc_te, y_em=None, y_pol=None, aspects=te_a)

            batch_size = info["batch_size"]
            epochs = info["epochs"]
            lr = LR
            patience = info["patience"]
            if "large" in model_key.lower():
                batch_size = max(4, batch_size // 2)
                lr = 1e-5

            # fp16 policy: DeBERTa -> disable fp16
            gpu_fp16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
            use_fp16 = bool(gpu_fp16_ok and ("deberta" not in model_key.lower()))

            out_dir = os.path.join(OUTPUT_ROOT, "exp1_multitask", model_key)
            os.makedirs(out_dir, exist_ok=True)

            model = BaseMultiTaskModel(
                model_name=info["name"],
                n_em=len(emotion2id),
                n_pol=len(pol2id),
                head_hidden=128,
                lambda_em=LAMBDA_EMO,
                lambda_pol=LAMBDA_POL,
                class_weights_em=class_weights_em_tensor,
                class_weights_pol=class_weights_pol_tensor
            ).to(DEVICE)

            trainer = DeterministicMultiTaskTrainer(
                model=model,
                args=make_args(out_dir, batch_size, info["grad_accum"], epochs, lr, use_fp16),
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                data_collator=collate,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
            )

            trainer.train()
            trainer.save_model(out_dir)
            tokenizer.save_pretrained(out_dir)

            pred = trainer.predict(ds_te_inputs)
            if isinstance(pred.predictions, dict):
                logits_em = pred.predictions["logits_em"]
            else:
                logits_em = pred.predictions[0]

            yhat_em = np.argmax(logits_em, axis=1)
            metrics = print_report(
                f"{model_key.upper()} MULTITASK - Emotion Test (BEST val checkpoint, NO polarity fed at test)",
                yte_e, yhat_em, id2emotion
            )
            all_results[model_key] = {"emotion_test_no_polarity_input": metrics}

            del model, trainer, tokenizer
            del enc_tr, enc_va, enc_te, ds_tr, ds_va, ds_te_inputs
            clear_memory_aggressive()

        except Exception as e:
            print(f"\n✗ ERROR processing {model_key}: {e}")
            traceback.print_exc()
            clear_memory_aggressive()
            continue

    return all_results


# EXPERIMENT 2: Train polarity only, Test polarity 

def run_experiment_2(tr_t, tr_a, tr_p, va_t, va_a, va_p, te_t, te_a, te_p,
                     pol2id, id2pol, class_weights_pol_tensor):
    print("\n" + "="*80)
    print("EXPERIMENT 2: TRAIN ON POLARITY ONLY, TEST ON POLARITY (NO POLARITY FED)")
    print("="*80)

    ytr_p = np.array([pol2id[x] for x in tr_p], dtype=np.int32)
    yva_p = np.array([pol2id[x] for x in va_p], dtype=np.int32)
    yte_p = np.array([pol2id[x] for x in te_p], dtype=np.int32)

    all_results = {}
    order = ["logistic_regression", "svm", "distilbert", "bert", "roberta",
             "distilroberta", "deberta", "bert-large", "roberta-large"]

    for model_key in order:
        if model_key not in MODELS:
            continue
        info = MODELS[model_key]
        print(f"\n{'='*80}\nEXPERIMENT 2: {model_key.upper()} ({info['name']})\n{'='*80}")
        clear_memory_aggressive()

        try:
            if info["type"] == "traditional_ml":
                extractor = MemoryEfficientFeatureExtractor("distilbert-base-uncased")
                Xtr = extractor.extract_features_batch(tr_t, tr_a, MAX_LENGTH, batch_size=8)
                Xte = extractor.extract_features_batch(te_t, te_a, MAX_LENGTH, batch_size=8)

                ml = TraditionalMLPolarityOnly(model_type=model_key, random_state=SEED, max_iter=1000)
                ml.fit(Xtr, ytr_p)
                yhat = ml.predict(Xte)

                metrics = print_report(
                    f"{model_key.upper()} (Polarity Only) - Polarity Test (NO polarity fed at test)",
                    yte_p, yhat, id2pol
                )
                all_results[model_key] = {"polarity_test_no_polarity_input": metrics}

                del extractor, Xtr, Xte, ml
                clear_memory_aggressive()
                continue

            tokenizer = load_tokenizer(model_key, info["name"])
            enc_tr = encode(tokenizer, tr_t, tr_a, MAX_LENGTH)
            enc_va = encode(tokenizer, va_t, va_a, MAX_LENGTH)
            enc_te = encode(tokenizer, te_t, te_a, MAX_LENGTH)

            ds_tr = EncDataset(enc_tr, y_pol=ytr_p, aspects=tr_a)
            ds_va = EncDataset(enc_va, y_pol=yva_p, aspects=va_a)

            # TEST: inputs only
            ds_te_inputs = EncDataset(enc_te, y_pol=None, aspects=te_a)

            batch_size = info["batch_size"]
            epochs = info["epochs"]
            lr = LR
            patience = info["patience"]
            if "large" in model_key.lower():
                batch_size = max(4, batch_size // 2)
                lr = 1e-5

            gpu_fp16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
            use_fp16 = bool(gpu_fp16_ok and ("deberta" not in model_key.lower()))

            out_dir = os.path.join(OUTPUT_ROOT, "exp2_polarity_only", model_key)
            os.makedirs(out_dir, exist_ok=True)

            model = BasePolarityOnlyModel(
                model_name=info["name"],
                n_pol=len(pol2id),
                head_hidden=128,
                class_weights=class_weights_pol_tensor
            ).to(DEVICE)

            trainer = DeterministicPolarityOnlyTrainer(
                model=model,
                args=make_args(out_dir, batch_size, info["grad_accum"], epochs, lr, use_fp16),
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                data_collator=collate,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
            )

            trainer.train()
            trainer.save_model(out_dir)
            tokenizer.save_pretrained(out_dir)

            pred = trainer.predict(ds_te_inputs)
            if isinstance(pred.predictions, dict):
                logits_pol = pred.predictions["logits_pol"]
            else:
                logits_pol = pred.predictions

            yhat_pol = np.argmax(logits_pol, axis=1)
            metrics = print_report(
                f"{model_key.upper()} POLARITY-ONLY - Polarity Test (BEST val checkpoint, NO polarity fed at test)",
                yte_p, yhat_pol, id2pol
            )
            all_results[model_key] = {"polarity_test_no_polarity_input": metrics}

            del model, trainer, tokenizer
            del enc_tr, enc_va, enc_te, ds_tr, ds_va, ds_te_inputs
            clear_memory_aggressive()

        except Exception as e:
            print(f"\n✗ ERROR processing {model_key}: {e}")
            traceback.print_exc()
            clear_memory_aggressive()
            continue

    return all_results


# EXPERIMENT 3: Train both tasks, Test polarity 

def run_experiment_3(tr_t, tr_a, tr_e, tr_p, va_t, va_a, va_e, va_p, te_t, te_a, te_e, te_p,
                     emotion2id, pol2id, id2pol, class_weights_em_tensor, class_weights_pol_tensor):
    print("\n" + "="*80)
    print("EXPERIMENT 3: TRAIN ON BOTH EMOTION & POLARITY, TEST ON POLARITY (NO POLARITY FED)")
    print("="*80)

    ytr_e = np.array([emotion2id[x] for x in tr_e], dtype=np.int32)
    yva_e = np.array([emotion2id[x] for x in va_e], dtype=np.int32)

    ytr_p = np.array([pol2id[x] for x in tr_p], dtype=np.int32)
    yva_p = np.array([pol2id[x] for x in va_p], dtype=np.int32)
    yte_p = np.array([pol2id[x] for x in te_p], dtype=np.int32)

    all_results = {}
    order = ["logistic_regression", "svm", "distilbert", "bert", "roberta",
             "distilroberta", "deberta", "bert-large", "roberta-large"]

    for model_key in order:
        if model_key not in MODELS:
            continue
        info = MODELS[model_key]
        print(f"\n{'='*80}\nEXPERIMENT 3: {model_key.upper()} ({info['name']})\n{'='*80}")
        clear_memory_aggressive()

        try:
            if info["type"] == "traditional_ml":
                extractor = MemoryEfficientFeatureExtractor("distilbert-base-uncased")
                Xtr = extractor.extract_features_batch(tr_t, tr_a, MAX_LENGTH, batch_size=8)
                Xte = extractor.extract_features_batch(te_t, te_a, MAX_LENGTH, batch_size=8)

                ml = TraditionalMLPolarityOnly(model_type=model_key, random_state=SEED, max_iter=1000)
                ml.fit(Xtr, ytr_p)
                yhat = ml.predict(Xte)

                metrics = print_report(
                    f"{model_key.upper()} (Exp3 baseline) - Polarity Test (NO polarity fed at test)",
                    yte_p, yhat, id2pol
                )
                all_results[model_key] = {"polarity_test_no_polarity_input": metrics}

                del extractor, Xtr, Xte, ml
                clear_memory_aggressive()
                continue

            tokenizer = load_tokenizer(model_key, info["name"])
            enc_tr = encode(tokenizer, tr_t, tr_a, MAX_LENGTH)
            enc_va = encode(tokenizer, va_t, va_a, MAX_LENGTH)
            enc_te = encode(tokenizer, te_t, te_a, MAX_LENGTH)

            ds_tr = EncDataset(enc_tr, y_em=ytr_e, y_pol=ytr_p, aspects=tr_a)
            ds_va = EncDataset(enc_va, y_em=yva_e, y_pol=yva_p, aspects=va_a)

            # TEST: inputs only
            ds_te_inputs = EncDataset(enc_te, y_em=None, y_pol=None, aspects=te_a)

            batch_size = info["batch_size"]
            epochs = info["epochs"]
            lr = LR
            patience = info["patience"]
            if "large" in model_key.lower():
                batch_size = max(4, batch_size // 2)
                lr = 1e-5

            gpu_fp16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
            use_fp16 = bool(gpu_fp16_ok and ("deberta" not in model_key.lower()))

            out_dir = os.path.join(OUTPUT_ROOT, "exp3_multitask_polarity_test", model_key)
            os.makedirs(out_dir, exist_ok=True)

            model = BaseMultiTaskModel(
                model_name=info["name"],
                n_em=len(emotion2id),
                n_pol=len(pol2id),
                head_hidden=128,
                lambda_em=LAMBDA_EMO,
                lambda_pol=LAMBDA_POL,
                class_weights_em=class_weights_em_tensor,
                class_weights_pol=class_weights_pol_tensor
            ).to(DEVICE)

            trainer = DeterministicMultiTaskTrainer(
                model=model,
                args=make_args(out_dir, batch_size, info["grad_accum"], epochs, lr, use_fp16),
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                data_collator=collate,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
            )

            trainer.train()
            trainer.save_model(out_dir)
            tokenizer.save_pretrained(out_dir)

            pred = trainer.predict(ds_te_inputs)
            if isinstance(pred.predictions, dict):
                logits_pol = pred.predictions["logits_pol"]
            else:
                logits_pol = pred.predictions[1]

            yhat_pol = np.argmax(logits_pol, axis=1)
            metrics = print_report(
                f"{model_key.upper()} MULTITASK - Polarity Test (BEST val checkpoint, NO polarity fed at test)",
                yte_p, yhat_pol, id2pol
            )
            all_results[model_key] = {"polarity_test_no_polarity_input": metrics}

            del model, trainer, tokenizer
            del enc_tr, enc_va, enc_te, ds_tr, ds_va, ds_te_inputs
            clear_memory_aggressive()

        except Exception as e:
            print(f"\n✗ ERROR processing {model_key}: {e}")
            traceback.print_exc()
            clear_memory_aggressive()
            continue

    return all_results


# PLOTS

def create_comparison_plots(exp1, exp2, exp3, output_dir):
    plt.style.use("seaborn-v0_8-darkgrid")

    all_models = set(exp1.keys()) | set(exp2.keys()) | set(exp3.keys())
    rows = []
    for m in sorted(all_models):
        rows.append({
            "model": m,
            "family": MODELS.get(m, {}).get("family", "Unknown"),
            "params": MODELS.get(m, {}).get("params", 0),
            "type": MODELS.get(m, {}).get("type", "Unknown"),
            "exp1_emotion_f1w": exp1.get(m, {}).get("emotion_test_no_polarity_input", {}).get("f1_weighted", 0),
            "exp1_emotion_f1m": exp1.get(m, {}).get("emotion_test_no_polarity_input", {}).get("f1_macro", 0),
            "exp2_pol_f1w": exp2.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_weighted", 0),
            "exp2_pol_f1m": exp2.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_macro", 0),
            "exp3_pol_f1w": exp3.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_weighted", 0),
            "exp3_pol_f1m": exp3.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_macro", 0),
        })
    df = pd.DataFrame(rows)
    df["sort_key"] = df["type"].apply(lambda x: 0 if x == "traditional_ml" else 1)
    df = df.sort_values(["sort_key", "params"])

    # --- Plot 1: overview 2x2
    fig1, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    d = df[df["exp1_emotion_f1w"] > 0]
    if len(d) > 0:
        x = np.arange(len(d))
        w = 0.35
        ax.bar(x - w/2, d["exp1_emotion_f1w"], w, label="F1 weighted", edgecolor="black")
        ax.bar(x + w/2, d["exp1_emotion_f1m"], w, label="F1 macro", edgecolor="black")
        ax.set_title("Exp1: Emotion test (best val checkpoint)")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in d["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    d = df[(df["exp2_pol_f1w"] > 0) | (df["exp3_pol_f1w"] > 0)]
    if len(d) > 0:
        x = np.arange(len(d))
        w = 0.35
        ax.bar(x - w/2, d["exp2_pol_f1w"], w, label="Exp2 (pol-only)", edgecolor="black", alpha=0.85)
        ax.bar(x + w/2, d["exp3_pol_f1w"], w, label="Exp3 (multi-task)", edgecolor="black", alpha=0.85)
        ax.set_title("Polarity test: Exp2 vs Exp3 (best val checkpoint)")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in d["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    fams = sorted(df["family"].unique().tolist())
    fam_rows = []
    for fam in fams:
        sub = df[df["family"] == fam]
        fam_rows.append({
            "family": fam,
            "exp1": sub["exp1_emotion_f1w"].mean(),
            "exp2": sub["exp2_pol_f1w"].mean(),
            "exp3": sub["exp3_pol_f1w"].mean(),
        })
    fam_df = pd.DataFrame(fam_rows)
    x = np.arange(len(fam_df))
    w = 0.25
    ax.bar(x - w, fam_df["exp1"], w, label="Exp1 emotion", edgecolor="black")
    ax.bar(x, fam_df["exp2"], w, label="Exp2 polarity", edgecolor="black")
    ax.bar(x + w, fam_df["exp3"], w, label="Exp3 polarity", edgecolor="black")
    ax.set_title("Average F1 by family")
    ax.set_xticks(x)
    ax.set_xticklabels(fam_df["family"], rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    for _, r in df.iterrows():
        if r["params"] > 0:
            ax.scatter(r["params"], r["exp1_emotion_f1w"], s=max(50, r["params"] * 2), edgecolor="black", alpha=0.7)
            ax.annotate(r["model"][:8], (r["params"], r["exp1_emotion_f1w"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Exp1 emotion vs model size")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("F1 weighted")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    fig1.suptitle("COMPREHENSIVE COMPARISON (NO POLARITY FED AT TEST INPUT)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment_comparison_overview.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- Plot 2: detailed side-by-side
    fig2, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    d = df[df["exp1_emotion_f1w"] > 0]
    if len(d) > 0:
        x = np.arange(len(d))
        ax.bar(x, d["exp1_emotion_f1w"], edgecolor="black", alpha=0.85)
        ax.scatter(x, d["exp1_emotion_f1m"], s=60, marker="D", edgecolor="black", zorder=5, label="F1 macro")
        ax.set_title("Exp1: Emotion test")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in d["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    d = df[df["exp2_pol_f1w"] > 0]
    if len(d) > 0:
        x = np.arange(len(d))
        ax.bar(x, d["exp2_pol_f1w"], edgecolor="black", alpha=0.85)
        ax.set_title("Exp2: Polarity test")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in d["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    d = df[df["exp3_pol_f1w"] > 0]
    if len(d) > 0:
        x = np.arange(len(d))
        ax.bar(x, d["exp3_pol_f1w"], edgecolor="black", alpha=0.85)
        ax.set_title("Exp3: Polarity test (multi-task)")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in d["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")

    fig2.suptitle("DETAILED RESULTS (NO POLARITY FED AT TEST INPUT)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_experiment_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # --- Plot 3: single vs multi task for polarity
    fig3, ax = plt.subplots(figsize=(14, 7))
    comp = df[(df["exp2_pol_f1w"] > 0) & (df["exp3_pol_f1w"] > 0)]
    if len(comp) > 0:
        x = np.arange(len(comp))
        w = 0.35
        ax.bar(x - w/2, comp["exp2_pol_f1w"], w, label="Single-task (Exp2)", edgecolor="black", alpha=0.85)
        ax.bar(x + w/2, comp["exp3_pol_f1w"], w, label="Multi-task (Exp3)", edgecolor="black", alpha=0.85)
        ax.set_title("Polarity: single-task vs multi-task (best val checkpoints)")
        ax.set_xticks(x)
        ax.set_xticklabels([s[:10] for s in comp["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        avg2 = comp["exp2_pol_f1w"].mean()
        avg3 = comp["exp3_pol_f1w"].mean()
        ax.text(
            0.02, 0.98,
            f"Avg F1 weighted:\nExp2: {avg2:.4f}\nExp3: {avg3:.4f}\nΔ: {avg3-avg2:+.4f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "single_vs_multi_task_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)


def main():
    clear_memory_aggressive()

    tr_t, tr_a, tr_e, tr_p = load_jsonl_aspect_level_emotion_polarity(TRAIN_FILE)
    va_t, va_a, va_e, va_p = load_jsonl_aspect_level_emotion_polarity(VALID_FILE)
    te_t, te_a, te_e, te_p = load_jsonl_aspect_level_emotion_polarity(TEST_FILE)

    emotion2id = make_label_map(tr_e)
    pol2id = make_label_map(tr_p)
    id2emotion = {v: k for k, v in emotion2id.items()}
    id2pol = {v: k for k, v in pol2id.items()}

    # Filter valid/test to labels seen in train
    def filter_to_seen(texts, aspects, emotions, pols):
        keep = [(e in emotion2id) and (p in pol2id) for e, p in zip(emotions, pols)]
        return (
            [t for t, k in zip(texts, keep) if k],
            [a for a, k in zip(aspects, keep) if k],
            [e for e, k in zip(emotions, keep) if k],
            [p for p, k in zip(pols, keep) if k],
        )

    va_t, va_a, va_e, va_p = filter_to_seen(va_t, va_a, va_e, va_p)
    te_t, te_a, te_e, te_p = filter_to_seen(te_t, te_a, te_e, te_p)

    # class weights
    class_weights_em_tensor = None
    try:
        ytr_e_tmp = np.array([emotion2id[x] for x in tr_e], dtype=np.int32)
        cw_em = compute_class_weight("balanced", classes=np.unique(ytr_e_tmp), y=ytr_e_tmp)
        class_weights_em_tensor = torch.tensor(cw_em, dtype=torch.float).to(DEVICE)
    except:
        pass

    class_weights_pol_tensor = None
    try:
        ytr_p_tmp = np.array([pol2id[x] for x in tr_p], dtype=np.int32)
        cw_pol = compute_class_weight("balanced", classes=np.unique(ytr_p_tmp), y=ytr_p_tmp)
        class_weights_pol_tensor = torch.tensor(cw_pol, dtype=torch.float).to(DEVICE)
    except:
        pass

    exp1 = run_experiment_1(
        tr_t, tr_a, tr_e, tr_p,
        va_t, va_a, va_e, va_p,
        te_t, te_a, te_e, te_p,
        emotion2id, pol2id, id2emotion,
        class_weights_em_tensor, class_weights_pol_tensor
    )

    exp2 = run_experiment_2(
        tr_t, tr_a, tr_p,
        va_t, va_a, va_p,
        te_t, te_a, te_p,
        pol2id, id2pol, class_weights_pol_tensor
    )

    exp3 = run_experiment_3(
        tr_t, tr_a, tr_e, tr_p,
        va_t, va_a, va_e, va_p,
        te_t, te_a, te_e, te_p,
        emotion2id, pol2id, id2pol,
        class_weights_em_tensor, class_weights_pol_tensor
    )

    # Save results
    all_results = {"experiment_1": exp1, "experiment_2": exp2, "experiment_3": exp3}
    with open(os.path.join(OUTPUT_ROOT, "all_experiment_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Summary CSV
    summary_rows = []
    for m in set(exp1.keys()) | set(exp2.keys()) | set(exp3.keys()):
        row = {"model": m}
        row.update(MODELS.get(m, {}))
        row["exp1_emotion_f1_weighted"] = exp1.get(m, {}).get("emotion_test_no_polarity_input", {}).get("f1_weighted", 0)
        row["exp1_emotion_f1_macro"] = exp1.get(m, {}).get("emotion_test_no_polarity_input", {}).get("f1_macro", 0)
        row["exp1_emotion_acc"] = exp1.get(m, {}).get("emotion_test_no_polarity_input", {}).get("acc", 0)

        row["exp2_polarity_f1_weighted"] = exp2.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_weighted", 0)
        row["exp2_polarity_f1_macro"] = exp2.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_macro", 0)
        row["exp2_polarity_acc"] = exp2.get(m, {}).get("polarity_test_no_polarity_input", {}).get("acc", 0)

        row["exp3_polarity_f1_weighted"] = exp3.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_weighted", 0)
        row["exp3_polarity_f1_macro"] = exp3.get(m, {}).get("polarity_test_no_polarity_input", {}).get("f1_macro", 0)
        row["exp3_polarity_acc"] = exp3.get(m, {}).get("polarity_test_no_polarity_input", {}).get("acc", 0)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ROOT, "experiment_summary.csv"), index=False)

    # Plots
    create_comparison_plots(exp1, exp2, exp3, OUTPUT_ROOT)

    print("\n DONE.")
    print(f"Results JSON: {os.path.join(OUTPUT_ROOT, 'all_experiment_results.json')}")
    print(f"Summary CSV : {os.path.join(OUTPUT_ROOT, 'experiment_summary.csv')}")
    print(f"Plots saved : {OUTPUT_ROOT}")
    print(" GUARANTEE: During TESTING, ONLY review+aspect are fed as model inputs. No polarity is fed.")

    return all_results


# RUN

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Missing required package: psutil. Install with: pip install psutil")
        raise

    try:
        results = main()
    except Exception as e:
        print(f"\n Experiment failed: {e}")
        traceback.print_exc()
        results = None

    clear_memory_aggressive()
    print("\n All experiments finished!")
