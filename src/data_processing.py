import pandas as pd
import json
import os
import nltk
from typing import List, Dict, Any
from datasets import load_dataset
import nlpaug.augmenter.word as naw
from deep_translator import GoogleTranslator  # Updated import

# Configure NLTK data path and download required resources
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

def ensure_nltk_resources():
    """
    Ensure NLTK resources are downloaded and available
    """
    try:
        nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        print(f"NLTK resources downloaded to {NLTK_DATA_PATH}")
        return True
    except Exception as e:
        print(f"Failed to download NLTK resources: {e}")
        return False

def load_huggingface_faq_data(dataset_name: str = "NebulaByte/E-Commerce_FAQs") -> List[Dict[str, Any]]:
    """
    Load FAQ data from Hugging Face datasets, cache locally
    """
    local_path = "data/ecommerce_faqs.json"
    if os.path.exists(local_path):
        print(f"Loading cached dataset from {local_path}")
        with open(local_path, 'r') as f:
            return json.load(f)
    
    print(f"Loading dataset {dataset_name} from Hugging Face...")
    try:
        dataset = load_dataset(dataset_name)
        faqs = [{
            "question": item["question"],
            "answer": item["answer"],
            "category": item.get("category", ""),
            "question_id": item.get("question_id", ""),
            "faq_url": item.get("faq_url", "")
        } for item in dataset["train"]]
        with open(local_path, 'w') as f:
            json.dump(faqs, f)
        print(f"Saved dataset to {local_path}, loaded {len(faqs)} FAQs")
        return faqs
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to local data...")
        return load_faq_data("data/faq_data.csv")

def load_faq_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load FAQ data from a local CSV or JSON file
    """
    print(f"Loading data from {file_path}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            faqs = df.to_dict('records')
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                faqs = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        print(f"Loaded {len(faqs)} FAQ entries")
        return faqs
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample dataset as fallback")
        sample_faqs = [
            {"question": "How do I track my order?", "answer": "You can track your order by logging into your account and visiting the Order History section."},
            {"question": "How do I reset my password?", "answer": "To reset your password, click on the 'Forgot Password' link on the login page."}
        ]
        return sample_faqs

def preprocess_faq(faqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess FAQ data: clean text, handle formatting, and filter invalid entries
    """
    processed_faqs = []
    for faq in faqs:
        # Safely handle question and answer fields
        question = faq.get('question')
        answer = faq.get('answer')
        
        # Convert to string and strip, handling None values
        question = str(question).strip() if question is not None else ""
        answer = str(answer).strip() if answer is not None else ""
        
        # Update FAQ dictionary
        faq['question'] = question
        faq['answer'] = answer
        
        # Only include FAQs with both question and answer
        if question and answer:
            processed_faqs.append(faq)
        else:
            print(f"Skipping invalid FAQ: question='{question}', answer='{answer}'")
    
    print(f"After preprocessing: {len(processed_faqs)} valid FAQ entries")
    return processed_faqs

def augment_faqs(faqs: List[Dict[str, Any]], max_faqs: int = 1000, enable_augmentation: bool = True) -> List[Dict[str, Any]]:
    """
    Augment FAQs with paraphrased questions if enabled
    """
    if not enable_augmentation:
        print("Augmentation disabled; returning original FAQs")
        return faqs
    
    if not ensure_nltk_resources():
        print("NLTK resources unavailable; skipping augmentation")
        return faqs
    
    aug = naw.SynonymAug()
    augmented = []
    for faq in faqs:
        augmented.append(faq)
        if len(augmented) < max_faqs:
            try:
                aug_question = aug.augment(faq['question'])[0]
                augmented.append({"question": aug_question, "answer": faq['answer'], "category": faq.get("category", "")})
            except Exception as e:
                print(f"Augmentation error for question '{faq['question'][:50]}...': {e}")
    print(f"Augmented to {len(augmented)} FAQs")
    return augmented

def translate_faq(faq: Dict[str, Any], target_lang: str = "es") -> Dict[str, Any]:
    """
    Translate FAQ to a target language using deep-translator
    """
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        translated = faq.copy()
        translated["question"] = translator.translate(faq["question"])
        translated["answer"] = translator.translate(faq["answer"])
        translated["language"] = target_lang
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return faq



