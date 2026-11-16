
from transformers import AutoTokenizer

TOKENIZERS = {
    "BPE - GPT-2": "gpt2",
    "WordPiece - Multilingual BERT": "bert-base-multilingual-cased",
    "SentencePiece - XLM-R": "xlm-roberta-base"
}

TEXTS = {
    "English": "Artificial intelligence is changing the world.",
    "Telugu": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
    "Hindi": "कृ त्रिम बुद्धिमत्ता दुनिया को बदल रही है।"
}

def main():
    loaded = {}

    for name, model in TOKENIZERS.items():
        print("Loading", name)
        loaded[name] = AutoTokenizer.from_pretrained(model)

    for language, text in TEXTS.items():
        print("\n==============================")
        print("LANGUAGE:", language)
        print("INPUT TEXT:", text)

        for name, tokenizer in loaded.items():
            encoded = tokenizer(text, add_special_tokens=True)
            ids = encoded["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)

            print("\n", name)
            print("Tokens:", tokens)
            print("Token IDs:", ids)
            print("Token count:", len(ids))

if __name__ == "__main__":
    main()
