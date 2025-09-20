#!/usr/bin/env python3
"""
Easy startup script for AI Game Server with provider detection
"""
import os
import sys
import logging
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.ai_apis.ai_provider_manager import AIProviderManager
from backend.server import app, logger

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                AI Game Server - Startup Script                ║
    ║          Pokemon Game Automation with Multiple AI Providers   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")

    env_vars = {
        'GEMINI_API_KEY': 'Google Gemini (Free tier available)',
        'OPENROUTER_API_KEY': 'OpenRouter (Paid)',
        'OPENAI_API_KEY': 'OpenAI API (Paid)',
        'OPENAI_ENDPOINT': 'OpenAI-compatible endpoint (e.g., LM Studio)',
        'NVIDIA_API_KEY': 'NVIDIA NIM (Paid)'
    }

    configured = []
    missing = []

    for var, description in env_vars.items():
        if os.environ.get(var):
            configured.append(f"✓ {description}")
        else:
            missing.append(f"✗ {description}")

    if configured:
        print("\n🟢 Configured providers:")
        for item in configured:
            print(f"   {item}")

    if missing:
        print("\n🔴 Missing providers:")
        for item in missing:
            print(f"   {item}")

    return len(configured) > 0

def check_local_providers():
    """Check for local AI providers"""
    print("\n🔍 Checking local AI providers...")

    local_providers = [
        {
            'name': 'LM Studio',
            'url': 'http://localhost:1234/v1',
            'description': 'Local model server (http://localhost:1234/v1)'
        },
        {
            'name': 'Ollama',
            'url': 'http://localhost:11434/api/tags',
            'description': 'Local model server (http://localhost:11434)'
        }
    ]

    available_local = []

    for provider in local_providers:
        try:
            import requests
            response = requests.get(provider['url'], timeout=5)
            if response.status_code == 200:
                available_local.append(provider)
                print(f"✓ {provider['name']} is running")
            else:
                print(f"✗ {provider['name']} not responding")
        except Exception:
            print(f"✗ {provider['name']} not running")

    return available_local

def setup_local_provider():
    """Setup environment for local provider"""
    print("\n🔧 Setting up local provider configuration...")

    # Check if LM Studio is available
    try:
        import requests
        response = requests.get('http://localhost:1234/v1/models', timeout=5)
        if response.status_code == 200:
            os.environ['OPENAI_API_KEY'] = 'not-needed'
            os.environ['OPENAI_ENDPOINT'] = 'http://localhost:1234/v1'
            os.environ['OPENAI_MODEL'] = 'local-model'
            print("✓ LM Studio detected and configured")
            return True
    except:
        pass

    # Check if Ollama is available
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            os.environ['OPENAI_API_KEY'] = 'not-needed'
            os.environ['OPENAI_ENDPOINT'] = 'http://localhost:11434/v1'
            os.environ['OPENAI_MODEL'] = 'llava'  # LLaVA for vision
            print("✓ Ollama detected and configured")
            return True
    except:
        pass

    return False

def test_providers():
    """Test all available providers"""
    print("\n🧪 Testing AI providers...")

    try:
        manager = AIProviderManager()
        status = manager.get_provider_status()
        available = manager.get_available_providers()

        print(f"\n📊 Provider Status:")
        for provider_name, provider_info in status.items():
            status_icon = "🟢" if provider_info['available'] else "🔴"
            print(f"   {status_icon} {provider_name}: {provider_info['status'].upper()}")
            if provider_info['error']:
                print(f"      Error: {provider_info['error']}")

        print(f"\n✅ Available providers: {len(available)}")
        for provider in available:
            print(f"   • {provider}")

        return len(available) > 0

    except Exception as e:
        print(f"❌ Error testing providers: {e}")
        return False

def show_usage():
    """Show usage instructions"""
    print("\n📖 Usage Instructions:")
    print("""
    1. API Endpoints:
       • GET  /api/status           - Get server status
       • GET  /api/providers/status - Get provider status
       • POST /api/ai-action        - Get AI action
       • POST /api/chat             - Chat with AI
       • GET  /api/screen           - Get game screen
       • GET  /api/stream           - Screen stream

    2. Web Interface:
       • Open http://localhost:5000 in your browser

    3. Example API calls:
       curl -X POST http://localhost:5000/api/ai-action \\
            -H "Content-Type: application/json" \\
            -d '{"goal": "Navigate through the game"}'

    4. Configuration:
       • Edit .env file to configure API keys
       • See AI_PROVIDER_SETUP.md for detailed setup
    """)

def show_provider_help():
    """Show help for setting up providers"""
    print("\n🛠️  Provider Setup Help:")
    print("""
    🌟 Google Gemini (Free):
       • Visit: https://makersuite.google.com/app/apikey
       • Get free API key
       • Set: GEMINI_API_KEY=your_key_here

    🌐 OpenRouter (Paid):
       • Visit: https://openrouter.ai/keys
       • Deposit funds and get API key
       • Set: OPENROUTER_API_KEY=your_key_here

    🏠 Local Models (LM Studio/Ollama):
       • LM Studio: Download from https://lmstudio.ai/
       • Ollama: Download from https://ollama.ai/
       • Set: OPENAI_ENDPOINT=http://localhost:1234/v1
       • Set: OPENAI_API_KEY=not-needed

    💡 Quick Start:
       • Install LM Studio and download a vision model
       • Start LM Studio server on port 1234
       • Run this script again
    """)

def main():
    """Main startup function"""
    print_banner()

    # Check if running in right directory
    if not Path('.env.example').exists():
        print("❌ Error: Please run this script from the ai-game-server directory")
        sys.exit(1)

    # Check environment
    has_env_config = check_environment()

    # Check local providers
    local_providers = check_local_providers()

    # Setup local provider if no cloud providers are configured
    if not has_env_config and not local_providers:
        print("\n🔄 No providers configured. Attempting to setup local provider...")
        if setup_local_provider():
            print("✅ Local provider configured successfully!")
        else:
            print("❌ Could not detect any local AI providers.")
            show_provider_help()
            return

    # Test providers
    has_working_providers = test_providers()

    # Show server information
    print("\n🚀 Starting AI Game Server...")
    print(f"   📍 Server will run on: http://localhost:5000")
    print(f"   📊 Status page: http://localhost:5000/api/status")
    print(f"   🔧 Provider status: http://localhost:5000/api/providers/status")

    if has_working_providers:
        print("   ✅ AI features are available!")
    else:
        print("   ⚠️  No AI providers available - manual actions only")

    # Show usage
    show_usage()

    print("\n" + "="*60)
    print("🎮 Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    # Start the server
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()