import time
import functools
from typing import Callable, Any, Dict, List
import torch
import psutil
import json
from evaluate import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def evaluate_response(generated_response: str, ground_truth: str = None) -> Dict[str, Any]:
    """
    Evaluate generated response with BLEU, ROUGE, and word overlap
    """
    results = {
        "length": len(generated_response),
        "word_count": len(generated_response.split())
    }
    
    if ground_truth:
        bleu = load("bleu")
        rouge = load("rouge")
        bleu_score = bleu.compute(predictions=[generated_response], references=[[ground_truth]])
        rouge_score = rouge.compute(predictions=[generated_response], references=[ground_truth])
        generated_words = set(generated_response.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        overlap = len(generated_words.intersection(ground_truth_words))
        results.update({
            "bleu": bleu_score["bleu"],
            "rouge": rouge_score["rougeL"],
            "word_overlap": overlap / len(ground_truth_words) if ground_truth_words else 0
        })
    
    return results

def evaluate_retrieval(embedder, test_set_path: str, k: int = 3) -> Dict[str, float]:
    """
    Evaluate retrieval quality with Precision@k and Recall@k
    """
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    
    precision, recall = [], []
    for item in test_set:
        query = item['query']
        true_ids = set(item['relevant_ids'])
        retrieved_faqs = embedder.retrieve_relevant_faqs(query, k)
        retrieved_ids = set(range(len(retrieved_faqs)))
        
        true_positives = len(true_ids & retrieved_ids)
        precision.append(true_positives / k if k > 0 else 0)
        recall.append(true_positives / len(true_ids) if true_ids else 0)
    
    return {
        "Precision@k": sum(precision) / len(precision) if precision else 0,
        "Recall@k": sum(recall) / len(recall) if recall else 0
    }

def baseline_keyword_search(query: str, faqs: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Keyword-based search baseline using TF-IDF
    """
    questions = [faq['question'] for faq in faqs]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [faqs[i] for i in top_k_indices]

def format_memory_stats():
    """
    Format memory usage statistics
    """
    system_stats = {
        "RAM": f"{psutil.virtual_memory().used / (1024 ** 3):.1f}GB / {psutil.virtual_memory().total / (1024 ** 3):.1f}GB",
        "RAM Usage": f"{psutil.virtual_memory().percent}%"
    }
    
    if torch.cuda.is_available():
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            gpu_stats[f"GPU {i}"] = f"{torch.cuda.get_device_name(i)}"
            gpu_stats[f"GPU {i} Memory"] = f"{torch.cuda.memory_allocated(i) / (1024 ** 3):.1f}GB / {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.1f}GB"
        system_stats.update(gpu_stats)
    
    return system_stats