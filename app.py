import streamlit as st
import time
import os
import gc
import torch
from src.data_processing import load_huggingface_faq_data, load_faq_data, preprocess_faq, augment_faqs
from src.embedding import FAQEmbedder
from src.llm_response import ResponseGenerator
from src.utils import time_function, format_memory_stats, evaluate_response, evaluate_retrieval, baseline_keyword_search
from deep_translator import GoogleTranslator  # Updated import

# Suppress CUDA warning and Torch path errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_NO_PATH_CHECK"] = "1"

st.set_page_config(page_title="E-Commerce FAQ Chatbot", layout="wide", initial_sidebar_state="expanded")

@time_function
def initialize_components(use_huggingface: bool = True, model_name: str = "microsoft/phi-2", enable_augmentation: bool = True):
    """
    Initialize RAG system components
    """
    try:
        if use_huggingface:
            faqs = load_huggingface_faq_data("NebulaByte/E-Commerce_FAQs")
        else:
            faqs = load_faq_data("data/faq_data.csv")
        
        processed_faqs = augment_faqs(preprocess_faq(faqs), enable_augmentation=enable_augmentation)
        embedder = FAQEmbedder()
        
        if os.path.exists("embeddings"):
            embedder.load("embeddings")
        else:
            embedder.create_embeddings(processed_faqs)
            embedder.save("embeddings")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        response_generator = ResponseGenerator(model_name=model_name)
        response_generator.generate_response("Warmup query", [{"question": "Test", "answer": "Test"}])
        
        return embedder, response_generator, len(processed_faqs)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        raise

def main():
    st.title("E-Commerce Customer Support FAQ Chatbot")
    st.subheader("Ask about orders, shipping, returns, or other e-commerce queries")
    
    st.sidebar.title("Configuration")
    use_huggingface = st.sidebar.checkbox("Use Hugging Face Dataset", value=True)
    enable_augmentation = st.sidebar.checkbox("Enable FAQ Augmentation", value=True, help="Generate paraphrased questions to expand dataset")
    target_lang = st.sidebar.selectbox("Language", ["en", "es", "fr"], index=0)
    
    model_options = {
        "Phi-2 (Recommended for 16GB RAM)": "microsoft/phi-2",
        "TinyLlama-1.1B (Fastest)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Mistral-7B (For 15GB+ GPU)": "mistralai/Mistral-7B-Instruct-v0.1"
    }
    selected_model = st.sidebar.selectbox("Select LLM Model", list(model_options.keys()), index=0)
    model_name = model_options[selected_model]
    
    if st.sidebar.checkbox("Show Memory Usage", value=True):
        st.sidebar.subheader("Memory Usage")
        for key, value in format_memory_stats().items():
            st.sidebar.text(f"{key}: {value}")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    
    if "system_initialized" not in st.session_state or st.sidebar.button("Reload System"):
        with st.spinner("Initializing system..."):
            try:
                st.session_state.embedder, st.session_state.response_generator, num_faqs = initialize_components(
                    use_huggingface=use_huggingface,
                    model_name=model_name,
                    enable_augmentation=enable_augmentation
                )
                st.session_state.system_initialized = True
                st.sidebar.success(f"System initialized with {num_faqs} FAQs!")
            except Exception as e:
                st.error(f"System initialization failed: {e}")
                return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        chat_container = st.container(height=400)
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                else:
                    st.markdown(f"**Bot**: {message['content']}")
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
        
        with st.form(key="chat_form"):
            user_query = st.text_input("Type your question:", key="user_input", placeholder="e.g., How do I track my order?")
            submit_button = st.form_submit_button("Ask")
        
        if len(st.session_state.chat_history) > 0:
            with st.form(key=f"feedback_form_{len(st.session_state.chat_history)}"):
                rating = st.slider("Rate this response (1-5)", 1, 5, key=f"rating_{len(st.session_state.chat_history)}")
                comments = st.text_area("Comments", key=f"comments_{len(st.session_state.chat_history)}")
                if st.form_submit_button("Submit Feedback"):
                    st.session_state.feedback.append({
                        "rating": rating,
                        "comments": comments,
                        "response": st.session_state.chat_history[-1]["content"]
                    })
                    with open("feedback.json", "w") as f:
                        json.dump(st.session_state.feedback, f)
                    st.success("Feedback submitted!")
    
    with col2:
        if st.session_state.get("system_initialized", False):
            st.subheader("Retrieved Information")
            info_container = st.container(height=500)
            with info_container:
                if "current_faqs" in st.session_state:
                    for i, faq in enumerate(st.session_state.current_faqs):
                        st.markdown(f"**Relevant FAQ #{i+1}**")
                        st.markdown(f"**Q**: {faq['question']}")
                        st.markdown(f"**A**: {faq['answer'][:150]}..." if len(faq['answer']) > 150 else f"**A**: {faq['answer']}")
                        st.markdown(f"*Similarity Score*: {faq['similarity']:.2f}")
                        if 'category' in faq and faq['category']:
                            st.markdown(f"*Category*: {faq['category']}")
                        st.markdown("---")
                else:
                    st.markdown("Ask a question to see relevant FAQs.")
    
    if "retrieval_time" in st.session_state and "generation_time" in st.session_state:
        st.sidebar.subheader("Performance Metrics")
        st.sidebar.markdown(f"Retrieval time: {st.session_state.retrieval_time:.2f} seconds")
        st.sidebar.markdown(f"Response generation: {st.session_state.generation_time:.2f} seconds")
        st.sidebar.markdown(f"Total time: {st.session_state.retrieval_time + st.session_state.generation_time:.2f} seconds")
    
    if submit_button and user_query:
        from src.data_processing import translate_faq
        translator = GoogleTranslator(source='auto', target='en')  # Updated translator
        if target_lang != "en":
            user_query_translated = translator.translate(user_query)
        else:
            user_query_translated = user_query
        
        if user_query_translated in st.session_state.query_cache:
            response, relevant_faqs = st.session_state.query_cache[user_query_translated]
        else:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_time = time.time()
            relevant_faqs = st.session_state.embedder.retrieve_relevant_faqs(user_query_translated)
            retrieval_time = time.time() - start_time
            
            if target_lang != "en":
                relevant_faqs = [translate_faq(faq, target_lang) for faq in relevant_faqs]
            
            start_time = time.time()
            response = st.session_state.response_generator.generate_response(user_query_translated, relevant_faqs)
            generation_time = time.time() - start_time
            
            if target_lang != "en":
                response = translator.translate(response, target=target_lang)
            
            st.session_state.query_cache[user_query_translated] = (response, relevant_faqs)
            st.session_state.retrieval_time = retrieval_time
            st.session_state.generation_time = generation_time
            st.session_state.current_faqs = relevant_faqs
        
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.query_cache = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if st.session_state.get("system_initialized", False):
        st.sidebar.subheader("Baseline Comparison")
        baseline_faqs = baseline_keyword_search(user_query_translated if 'user_query_translated' in locals() else "", st.session_state.embedder.faqs)
        st.sidebar.write(f"RAG FAQs: {[faq['question'][:50] for faq in st.session_state.get('current_faqs', [])]}")
        st.sidebar.write(f"Keyword FAQs: {[faq['question'][:50] for faq in baseline_faqs]}")
    
    st.subheader("Sample Questions")
    sample_questions = [
        "How do I track my order?",
        "What should I do if my delivery is delayed?",
        "How do I return a product?",
        "Can I cancel my order after placing it?",
        "How quickly will my order be delivered?"
    ]
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col_idx = i % 2
        if cols[col_idx].button(question, key=f"sample_{i}"):
            st.session_state.user_input = question
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            translator = GoogleTranslator(source='auto', target='en')  # Updated translator
            if target_lang != "en":
                question_translated = translator.translate(question)
            else:
                question_translated = question
            
            if question_translated in st.session_state.query_cache:
                response, relevant_faqs = st.session_state.query_cache[question_translated]
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                start_time = time.time()
                relevant_faqs = st.session_state.embedder.retrieve_relevant_faqs(question_translated)
                retrieval_time = time.time() - start_time
                
                if target_lang != "en":
                    relevant_faqs = [translate_faq(faq, target_lang) for faq in relevant_faqs]
                
                start_time = time.time()
                response = st.session_state.response_generator.generate_response(question_translated, relevant_faqs)
                generation_time = time.time() - start_time
                
                if target_lang != "en":
                    response = translator.translate(response, target=target_lang)
                
                st.session_state.query_cache[question_translated] = (response, relevant_faqs)
                st.session_state.retrieval_time = retrieval_time
                st.session_state.generation_time = generation_time
                st.session_state.current_faqs = relevant_faqs
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


