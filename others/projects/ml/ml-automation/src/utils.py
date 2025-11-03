def get_metric_for_backend(metric_name, backend):
    """
    Преобразует название метрики в формат, понятный бэкенду.
    """
    mapping = {
        "flaml": {
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "mae": "mae",
            "mse": "mse",
            "r2": "r2"
        },
        "h2o": {
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "auc",  # в H2O это просто 'auc'
            "mae": "mae",
            "mse": "mse",
            "r2": "r2"
        },
        "autosklearn": {
            # autosklearn использует sklearn.metrics.SCORERS
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "mae": "mean_absolute_error",
            "mse": "mean_squared_error",
            "r2": "r2"
        }
    }
    return mapping[backend].get(metric_name, metric_name)


def simple_text_preprocess(series):
    """Базовая предобработка текста: lowercase, удаление лишних пробелов.

    Возвращает Series очищенных строк.
    """
    try:
        return series.fillna("").astype(str).str.lower().str.strip()
    except Exception:
        return series


def embed_texts(texts, model_name_or_path=None):
    """Попытка получить эмбеддинги для списка текстов.

    Сначала пытается использовать sentence-transformers (если установлено).
    Если его нет, возвращает None.
    """
    try:
        # lazy import to avoid heavy deps when not needed
        from sentence_transformers import SentenceTransformer
        model_name = model_name_or_path or "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        emb = model.encode(list(texts), show_progress_bar=False)
        return emb
    except Exception:
        try:
            # fallback to transformers + pooling
            from transformers import AutoTokenizer, AutoModel
            import torch

            model_name = model_name_or_path or "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked = outputs.last_hidden_state * attention_mask
            summed = masked.sum(1)
            counts = attention_mask.sum(1).clamp(min=1)
            emb = (summed / counts).cpu().numpy()
            return emb
        except Exception:
            return None