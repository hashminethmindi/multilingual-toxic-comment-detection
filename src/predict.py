import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/xlm_final/best_model"

MAX_LENGTH = 128


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------

print("\nLoading XLM-R model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)

model.eval()

print("Model loaded successfully.\n")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_toxicity(text):

    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    non_toxic_probability = (
        probabilities[0][0].item()
    )

    toxic_probability = (
        probabilities[0][1].item()
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    if prediction == 1:
        predicted_label = "TOXIC"
        confidence = toxic_probability

    else:
        predicted_label = "NON-TOXIC"
        confidence = non_toxic_probability

    return {
        "label": prediction,
        "prediction": predicted_label,
        "confidence": confidence,
        "non_toxic_probability": non_toxic_probability,
        "toxic_probability": toxic_probability
    }


# --------------------------------------------------
# Interactive testing
# --------------------------------------------------

def main():

    print(
        "Multilingual Toxic Comment Detection"
    )

    print(
        "------------------------------------"
    )

    print(
        "Enter comments in English, Sinhala, "
        "Tamil, Singlish or Tanglish."
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        text = input(
            "Comment: "
        ).strip()

        if text.lower() == "exit":

            print(
                "\nPrediction program stopped."
            )

            break

        if not text:

            print(
                "Please enter a comment.\n"
            )

            continue

        result = predict_toxicity(
            text
        )

        print(
            "\nPrediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            "Non-toxic probability:",
            f"{result['non_toxic_probability'] * 100:.2f}%"
        )

        print(
            "Toxic probability:",
            f"{result['toxic_probability'] * 100:.2f}%"
        )

        print()


if __name__ == "__main__":
    main()