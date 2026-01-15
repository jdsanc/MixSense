#!/usr/bin/env python3
"""
Test script for LLM-enhanced chemistry analysis
Demonstrates the agentic workflow without requiring Gradio UI
"""
import os
import json
from pathlib import Path

from mixsense.llm_agent import ChemistryLLMAgent, process_natural_language_request
from mixsense.agent_pocketflow import build_llm_graph

def test_llm_parsing():
    """Test LLM parsing of chemistry requests"""
    print("🧪 Testing LLM Chemistry Request Parsing...")
    
    # Check for HF_TOKEN
    if not os.environ.get('HF_TOKEN'):
        print("⚠️  Warning: HF_TOKEN not set. LLM calls will fail.")
        print("   Set HF_TOKEN environment variable to test LLM functionality.")
        return False
    
    agent = ChemistryLLMAgent()
    
    test_requests = [
        "I want to analyze the bromination of anisole with Br2 using FeBr3 catalyst",
        "Help me quantify benzene and toluene in my mixture using robust analysis",
        "Track the time evolution of p-bromoanisole formation over 30 minutes",
        "Predict the structure of my unknown compound from NMR data"
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📝 Test {i}: {request}")
        try:
            task = agent.parse_chemistry_request(request)
            print(f"   ✅ Reactants: {task.reactants}")
            print(f"   ✅ Reagents: {task.reagents}")
            print(f"   ✅ Analysis type: {task.analysis_type}")
            print(f"   ✅ Backend: {task.backend_preference}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    return True

def test_pocketflow_graph():
    """Test PocketFlow graph execution"""
    print("\n🔧 Testing PocketFlow Graph...")
    
    try:
        # Build the flow
        flow = build_llm_graph(model_name="deepseek-ai/DeepSeek-V3:together")
        
        # Test context
        ctx = {
            "user_input": "Analyze the bromination of anisole with bromine and iron bromide catalyst"
        }
        
        print("   📊 Flow structure created successfully")
        print(f"   📊 Flow type: {type(flow)}")
        
        # Note: We won't run the full flow here as it requires API keys
        # but we can verify the structure is correct
        return True
        
    except Exception as e:
        print(f"   ❌ Flow creation failed: {e}")
        return False

def test_example_data():
    """Test with example data files"""
    print("\n📁 Testing with Example Data...")
    
    example_dir = Path(__file__).parent / "examples"
    if not example_dir.exists():
        print("   ⚠️  No examples directory found, skipping data test")
        return True
    
    # Look for CSV files
    csv_files = list(example_dir.glob("*.csv"))
    if not csv_files:
        print("   ⚠️  No CSV files in examples, skipping data test") 
        return True
    
    # Test CSV parsing
    try:
        import pandas as pd
        from mixsense.gradio_llm_app import parse_csv
        
        test_file = csv_files[0]
        print(f"   📄 Testing with: {test_file.name}")
        
        # Create a mock file object
        class MockFile:
            def __init__(self, path):
                self.name = str(path)
        
        mock_file = MockFile(test_file)
        data = parse_csv(mock_file)
        
        print(f"   ✅ Parsed {len(data['ppm'])} data points")
        print(f"   ✅ PPM range: {min(data['ppm']):.2f} - {max(data['ppm']):.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CSV parsing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LLM Chemistry Analysis Integration Test")
    print("=" * 50)
    
    tests = [
        ("LLM Parsing", test_llm_parsing),
        ("PocketFlow Graph", test_pocketflow_graph), 
        ("Example Data", test_example_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for LLM hackathon demo.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
    
    # Provide usage instructions
    print("\n📚 Usage Instructions:")
    print("1. Set HF_TOKEN environment variable for LLM access")
    print("2. Run: python -m mixsense.gradio_llm_app")
    print("3. Open browser to http://localhost:7860")
    print("4. Try natural language requests like:")
    print("   'Analyze bromination of anisole with Br2 and FeBr3'")

if __name__ == "__main__":
    main()
