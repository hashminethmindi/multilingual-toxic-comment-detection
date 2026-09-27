## Trained XLM-RoBERTa Model

The final fine-tuned multilingual toxic comment detection model is hosted on Hugging Face.

**Model:** `Hashmi2004/multilingual-toxic-comment-xlm-roberta`

Final test results:

- Accuracy: 84.50%
- Precision: 79.83%
- Recall: 78.21%
- F1-score: 79.01%

The model supports:

- English
- Sinhala
- Tamil
- Singlish
- Tanglish

### Load the model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Hashmi2004/multilingual-toxic-comment-xlm-roberta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)