import sys
import os
from pathlib import Path
import subprocess

def check_environment():
    """Check if we're in the correct conda environment"""
    env_name = "llm"
    current_env = Path(sys.executable).parts[-3]
    
    if current_env != env_name:
        raise EnvironmentError(
            f"Not in {env_name} environment. Current environment: {current_env}\n"
            "Please run:\n"
            "1. conda env create -f environment.yml\n"
            f"2. conda activate {env_name}"
        )
    return True

class LlamaTool:
    def __init__(self):
        """Initialize the Llama tool with environment check"""
        check_environment()
        # Add your initialization code here
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the Llama model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Configure model settings
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model with GPU support if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            print(f"Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, prompt, max_length=2000):
        """Generate a response using the loaded model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Encode the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode and return the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self.tokenizer = None
            print("Cleanup completed successfully")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def main():
    """Example usage of the LlamaTool"""
    tool = LlamaTool()
    if tool.load_model():
        try:
            # Example prompt
            prompt = "What are zeolites used for?"
            response = tool.generate_response(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        finally:
            tool.cleanup()

if __name__ == "__main__":
    main()
