from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict, Any
import gc
import psutil

class ResponseGenerator:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """
        Initialize the response generator with an LLM
        """
        print(f"Loading LLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                available_memory = psutil.virtual_memory().total / (1024 ** 3)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                max_memory = {0: f"{min(gpu_memory, 15)}GiB", "cpu": f"{min(available_memory, 30)}GiB"}
                print(f"Setting max_memory: {max_memory}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    max_memory=max_memory,
                    offload_folder="offload",
                    offload_state_dict=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": "cpu"},
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
        except Exception as e:
            print(f"Model loading error: {e}")
            print("Falling back to TinyLlama...")
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": self.device},
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        
        print("LLM loaded successfully")
    
    def generate_response(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with retrieved FAQs as context
        """
        prompt = self._create_prompt(query, relevant_faqs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return response
    
    def _create_prompt(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the LLM with retrieved FAQs as context
        """
        faq_context = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in relevant_faqs])
        prompt = f"""
Below are some relevant e-commerce customer support FAQ entries:

{faq_context}

Based on the information above, provide a helpful, accurate, and concise response to the following customer query:
Customer Query: {query}

Response:
"""
        return prompt