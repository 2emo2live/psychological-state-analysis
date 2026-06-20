import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

# Настройки
TRITON_URL = "localhost:8000"
MODEL_NAME = "psychological_state_analyzer"
MAX_LEN = 512
TOKENIZER_NAME = "distilbert-base-uncased"

# Подготовка токенизатора
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def preprocess_text(text: str):
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="np",
    )
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }


def infer(text: str) -> int:
    # Подготовка данных
    data = preprocess_text(text)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    # Создание клиента
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # Формирование входных тензоров
    triton_inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    triton_inputs[0].set_data_from_numpy(input_ids)
    triton_inputs[1].set_data_from_numpy(attention_mask)

    # Запрос
    outputs = [httpclient.InferRequestedOutput("logits")]
    response = client.infer(MODEL_NAME, triton_inputs, outputs=outputs)
    logits = response.as_numpy("logits")
    predicted_class = int(np.argmax(logits, axis=1)[0])
    return predicted_class


if __name__ == "__main__":
    text = "I feel very anxious and worried"
    pred = infer(text)
    print(f"Predicted class index: {pred}")
