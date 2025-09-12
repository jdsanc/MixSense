import os
import json
import requests
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

# Setup logging
from datetime import datetime
import sys

# Create outputs directory if it doesn't exist
output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)

# Setup logging to both file and console
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = output_dir / f"nmr_analysis_{timestamp}.out"

class TeeStream:
    def __init__(self, original_stream, file_stream):
        self.original_stream = original_stream
        self.file_stream = file_stream

    def write(self, data):
        self.original_stream.write(data)
        self.file_stream.write(data)
        self.file_stream.flush()
        self.original_stream.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()

# Setup logging with both file and console output
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see more details
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Tee stdout and stderr to the log file
log_file_stream = open(log_file, 'a')
sys.stdout = TeeStream(sys.stdout, log_file_stream)
sys.stderr = TeeStream(sys.stderr, log_file_stream)

logger.info(f"Starting NMR Analysis Tool - Logging to {log_file}")
logger.info("Using Hugging Face Inference API")

@dataclass
class NMRPeak:
    """Representation of an NMR peak"""
    chemical_shift: float
    intensity: float
    multiplicity: Optional[str] = None
    coupling_constant: Optional[float] = None
    nucleus_type: str = "1H"  # Can be "1H", "13C", etc.

@dataclass
class NMRSpectrum:
    """Representation of an NMR spectrum"""
    peaks: List[NMRPeak]
    nucleus_type: str  # "1H" or "13C"
    solvent: Optional[str] = None
    frequency: Optional[float] = None
    temperature: Optional[float] = None

class NMRAnalysisTool:
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the NMR Analysis Tool
        
        Args:
            hf_token: HuggingFace access token for Llama-3
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HuggingFace token not provided. Set HF_TOKEN environment variable or pass token.")
        
        self.model = None
        self.sampling_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Llama-3 model using Hugging Face Inference API"""
        try:
            # Start with a public model to verify token works
            self.model_name = "gpt2"  # Testing with a non-gated model first
            self.url = "https://api-inference.huggingface.co/models/gpt2"  # Public model
            self.headers = {
                'Authorization': f'Bearer {self.hf_token}',
                'Content-Type': 'application/json'
            }
            self.headers = {
                'Authorization': f'Bearer {self.hf_token}',
                'Content-Type': 'application/json'
            }
            
            # Define optimized generation parameters
            self.generation_kwargs = {
                "max_new_tokens": 5000,    # Allow longer responses for detailed analysis
                "temperature": 0.01,       # More deterministic for scientific analysis
                "top_k": 50,              # Balance between diversity and focus
                "top_p": 0.95,            # Slightly higher for technical accuracy
                "return_full_text": False,  # Only return the generated response
                "repetition_penalty": 1.15, # Slightly higher to prevent repetition
                "do_sample": True          # Enable sampling for more natural responses
            }
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def parse_nmr_json(self, json_data: Union[str, Dict]) -> List[NMRSpectrum]:
        """Parse NMR JSON data into structured format"""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        spectra = []
        for spectrum_data in data.get("spectra", []):
            peaks = [
                NMRPeak(
                    chemical_shift=peak["shift"],
                    intensity=peak["intensity"],
                    multiplicity=peak.get("multiplicity"),
                    coupling_constant=peak.get("j_coupling"),
                    nucleus_type=spectrum_data.get("nucleus", "1H")
                )
                for peak in spectrum_data.get("peaks", [])
            ]
            
            spectrum = NMRSpectrum(
                peaks=peaks,
                nucleus_type=spectrum_data.get("nucleus", "1H"),
                solvent=spectrum_data.get("solvent"),
                frequency=spectrum_data.get("frequency"),
                temperature=spectrum_data.get("temperature")
            )
            spectra.append(spectrum)
            
        return spectra

    async def predict_products(self, reactants: List[str]) -> List[str]:
        """Placeholder for reaction product prediction"""
        # This will be replaced by your actual tool
        logger.info("Product prediction tool not implemented yet")
        return []

    async def get_reference_spectra(self, smiles: List[str]) -> List[NMRSpectrum]:
        """Placeholder for reference spectra retrieval"""
        # This will be replaced by your actual tool
        logger.info("Spectra retrieval tool not implemented yet")
        return []

    async def analyze_spectrum(self, 
                             input_spectrum: List[NMRSpectrum],
                             reference_spectra: List[NMRSpectrum]) -> Dict:
        """Placeholder for spectrum analysis"""
        # This will be replaced by your actual tool
        logger.info("Spectrum analysis tool not implemented yet")
        return {}

    def test_api(self, query: str) -> str:
        """Test the API connection with a simple query"""
        try:
            logger.info(f"Testing API with query: {query}")
            
            # Format prompt with Llama 2 chat format
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

{query} [/INST]"""
            
            # Prepare the payload
            payload = {
                "inputs": formatted_prompt,
                **self.generation_kwargs  # Include parameters directly in the root
            }
            
            # Make API request
            logger.info("Sending request to API...")
            logger.debug(f"Request URL: {self.url}")
            logger.debug(f"Request headers: {self.headers}")
            logger.debug(f"Request payload: {payload}")
            
            response = requests.post(self.url, headers=self.headers, json=payload)
            
            # Log the raw response for debugging
            logger.info(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Raw response: {response.text}")
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            logger.debug(f"Parsed response: {response_json}")
            
            if isinstance(response_json, list) and len(response_json) > 0:
                generated_text = response_json[0].get('generated_text', '')
                if generated_text:
                    return generated_text.strip()
                else:
                    raise ValueError("No generated text in response")
            else:
                raise ValueError(f"Unexpected response format: {response_json}")
                
        except Exception as e:
            logger.error(f"API test failed: {str(e)}")
            return f"Error: {str(e)}"

    async def process_query(self, query: str, nmr_data: Dict) -> str:
        """Process a user query about NMR spectrum"""
        try:
            # Parse input spectrum
            input_spectra = self.parse_nmr_json(nmr_data)
            
            # Extract reactant information from query using LLM
            prompt = f"""Given this query about an NMR spectrum: "{query}"
            Extract the reactants mentioned. Output them as a comma-separated list."""
            
            # Format prompt with Llama 2 chat format
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are an expert NMR spectroscopy analyst. You accurately analyze and interpret NMR spectra.
<</SYS>>

{prompt} [/INST]"""
            
            # Make API request for reactants
            payload = {
                "inputs": formatted_prompt,
                "parameters": self.generation_kwargs
            }
            response = requests.post(self.url, headers=self.headers, json=payload)
            
            # Check if the request was successful
            response.raise_for_status()
            
            try:
                response_json = response.json()
                logger.debug(f"API Response: {response_json}")
                
                if isinstance(response_json, list) and len(response_json) > 0:
                    generated_text = response_json[0].get('generated_text', '')
                    if generated_text:
                        reactants = [r.strip() for r in generated_text.split(",") if r.strip()]
                    else:
                        raise ValueError("No generated text in response")
                else:
                    raise ValueError(f"Unexpected response format: {response_json}")
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response. Status code: {response.status_code}, Content: {response.text}")
                raise ValueError(f"Invalid JSON response from API: {str(e)}")
            
            # Predict possible products
            predicted_products = await self.predict_products(reactants)
            
            # Get reference spectra
            reference_spectra = await self.get_reference_spectra(predicted_products)
            
            # Analyze spectra
            analysis_results = await self.analyze_spectrum(input_spectra, reference_spectra)
            
            # Generate final response
            result_prompt = f"""<s>[INST] <<SYS>>
You are an expert NMR spectroscopy analyst providing detailed analysis results.
<</SYS>>

Based on these NMR analysis results:
Input spectrum: ```{input_spectra}```
Predicted products: ```{predicted_products}```
Analysis results: ```{analysis_results}```

Provide a natural language explanation of:
1. What products were found
2. Their approximate percentages
3. The confidence in this analysis
4. Any notable observations about the spectrum [/INST]"""
            
            # Make API request for final analysis
            payload = {
                "inputs": result_prompt,
                "parameters": self.generation_kwargs
            }
            response = requests.post(self.url, headers=self.headers, json=payload)
            
            # Check if the request was successful
            response.raise_for_status()
            
            try:
                response_json = response.json()
                logger.debug(f"Final Analysis Response: {response_json}")
                
                if isinstance(response_json, list) and len(response_json) > 0:
                    generated_text = response_json[0].get('generated_text', '')
                    if generated_text:
                        return generated_text.strip()
                    else:
                        raise ValueError("No generated text in final response")
                else:
                    raise ValueError(f"Unexpected final response format: {response_json}")
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response. Status code: {response.status_code}, Content: {response.text}")
                raise ValueError(f"Invalid JSON response from API: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

def main():
    """Example usage"""
    import asyncio
    
    logger.info("Starting API test...")
    
    # Initialize tool
    logger.info("Initializing Tool...")
    try:
        tool = NMRAnalysisTool(hf_token=os.environ.get("HF_TOKEN"))
        
        # Test simple query first
        test_query = "Where is India?"
        logger.info("Testing simple query first...")
        result = tool.test_api(test_query)
        logger.info("Test Result:")
        logger.info(result)
        
        if "Error" in result:
            logger.error("API test failed, skipping NMR analysis")
            return
            
        logger.info("API test successful, proceeding with NMR analysis...")
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return
    
    # Example NMR data
    example_data = {
        "spectra": [{
            "nucleus": "1H",
            "solvent": "CDCl3",
            "frequency": 400,
            "peaks": [
                {"shift": 7.26, "intensity": 1.0, "multiplicity": "s"},
                {"shift": 3.72, "intensity": 2.1, "multiplicity": "t", "j_coupling": 7.2}
            ]
        }]
    }
    
    try:
        # Initialize tool
        logger.info("Initializing NMR Analysis Tool...")
        tool = NMRAnalysisTool(hf_token=os.environ.get("HF_TOKEN"))
        
        # Example query
        query = "Hey chat, I mixed together ethanol and acetic acid and get this crude NMR spectrum. What's in there?"
        logger.info(f"Processing query: {query}")
        
        # Run analysis
        result = asyncio.run(tool.process_query(query, example_data))
        logger.info("Analysis completed successfully")
        logger.info("Result:")
        logger.info(result)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Analysis session completed")
        if 'log_file_stream' in globals():
            log_file_stream.close()

if __name__ == "__main__":
    main()
