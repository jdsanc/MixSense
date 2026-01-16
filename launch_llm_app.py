#!/usr/bin/env python3
"""
Launch script for LLM-Enhanced Chemistry Analysis App
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress HuggingFace tokenizers parallelism warning
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_environment():
    """Check if environment is properly configured"""
    issues = []
    
    # Check HF_TOKEN
    if not os.environ.get('HF_TOKEN'):
        issues.append("HF_TOKEN environment variable not set (required for LLM access)")
    
    # Check Python packages
    required_packages = [
        'gradio', 'pandas', 'numpy', 'requests', 'rdkit', 'pocketflow'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing required package: {package}")
    
    return issues

def main():
    print("🧪 LLM-Enhanced Chemistry Analysis App")
    print("=" * 50)
    
    # Check environment
    issues = check_environment()
    if issues:
        print("❌ Environment Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Setup Instructions:")
        print("   1. Install packages: pip install gradio pandas numpy requests rdkit pocketflow")
        print("   2. Set HF_TOKEN: export HF_TOKEN=your_huggingface_token")
        print("   3. Re-run this script")
        return
    
    print("✅ Environment check passed!")
    print("\n🚀 Starting LLM Chemistry Analysis App...")
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Import and launch the app
        from app.gradio_llm_app import demo
        
        print("🌐 App will be available at: http://localhost:7860")
        print("📱 Gradio will also provide a shareable public URL")
        print("\n💬 Try these example requests:")
        print("   • 'Analyze bromination of anisole with Br2 and FeBr3'")
        print("   • 'Help me quantify benzene and toluene in my mixture'")
        print("   • 'Track p-bromoanisole formation over time'")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        # Launch with public sharing enabled
        demo.launch(
            share=True,
            server_name="0.0.0.0", 
            server_port=7860,
            show_api=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start app: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check that all dependencies are installed")
        print("   • Verify HF_TOKEN is set correctly")
        print("   • Try running: python app/test_llm_integration.py")

if __name__ == "__main__":
    main()
