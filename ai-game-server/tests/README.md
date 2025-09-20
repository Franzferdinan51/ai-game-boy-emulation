# AI Provider Test Suite

This directory contains comprehensive tests for AI provider integration in the PyGB AI Game Server.

## Test Files

### Core Provider Tests
- **`test_ai_api_base_comprehensive.py`** - Base class functionality and abstract methods
- **`test_gemini_api.py`** - Google Gemini API integration tests
- **`test_openrouter_api.py`** - OpenRouter API integration tests
- **`test_nvidia_api.py`** - NVIDIA NIM API integration tests
- **`test_openai_compatible_api.py`** - OpenAI-compatible API tests

### Integration Tests
- **`test_ai_action_pipeline.py`** - Complete AI action execution pipeline
- **`test_screen_capture_ai.py`** - Screen capture and AI analysis integration
- **`test_chat_functionality.py`** - Comprehensive chat functionality tests

### Infrastructure Tests
- **`test_error_handling.py`** - Error handling and fallback mechanisms
- **`test_environment_configuration.py`** - Environment variable configuration

## Running Tests

### Using the Test Runner
```bash
# From PyGB root directory
python run_ai_provider_tests.py --all
python run_ai_provider_tests.py --provider gemini
python run_ai_provider_tests.py --category integration
```

### Using pytest directly
```bash
cd ai-game-server
pytest tests/ -v
pytest tests/test_gemini_api.py -v
pytest tests/ -k "integration" -v
```

### Environment Variables Required

Set these environment variables for full testing:

```bash
# API Keys
export GEMINI_API_KEY="your-gemini-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export NVIDIA_API_KEY="your-nvidia-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# API Endpoints
export NVIDIA_BASE_URL="http://localhost:8000"
export OPENAI_BASE_URL="http://localhost:1234"
export LM_STUDIO_URL="http://localhost:1234"
```

## Test Coverage

### Provider Coverage
- ✅ Google Gemini API
- ✅ OpenRouter API
- ✅ NVIDIA NIM API
- ✅ OpenAI-compatible APIs (LM Studio, local servers)

### Integration Coverage
- ✅ Screen capture and analysis
- ✅ Action execution pipeline
- ✅ Error handling and recovery
- ✅ Performance benchmarks
- ✅ Chat functionality
- ✅ Environment configuration

### Error Scenarios
- ✅ Network timeouts
- ✅ API rate limiting
- ✅ Invalid responses
- ✅ Authentication failures
- ✅ Resource constraints

## Performance Testing

```bash
# Run performance benchmarks
python run_ai_provider_tests.py --benchmark
```

Metrics measured:
- Response time
- Throughput (actions/second)
- Memory usage
- Error rates
- Concurrency performance

## Test Categories

### Unit Tests
Individual component testing with mocked dependencies.

### Integration Tests
Full pipeline testing with realistic scenarios.

### Error Tests
Failure scenario testing and recovery validation.

### Performance Tests
Load and stress testing with metrics collection.

## Mock Objects

The test suite uses comprehensive mock objects:
- **MockEmulator**: Simulates Game Boy emulator behavior
- **MockAIAPI**: Simulates AI API responses
- **MockScreen**: Simulates screen capture scenarios
- **ErrorTestProvider**: Simulates various error conditions

## Continuous Integration

The test suite is designed for CI/CD integration:
- JUnit XML output support
- JSON report generation
- HTML report generation
- Performance metric collection

## Contributing

### Adding New Provider Tests
1. Create `test_new_provider.py`
2. Implement required test methods
3. Update `run_ai_provider_tests.py`
4. Update documentation

### Test Standards
- Use descriptive test names
- Include docstrings
- Follow AAA pattern (Arrange-Act-Assert)
- Mock external dependencies
- Clean up after tests

## Troubleshooting

### Common Issues
1. **Import errors**: Check Python path and installation
2. **Environment variables**: Verify API keys and endpoints
3. **Network errors**: Check connectivity and API status
4. **Test failures**: Review detailed error messages

### Debug Mode
```bash
python run_ai_provider_tests.py --all --verbose
```

For more detailed information, see the [AI Provider Testing Guide](../../AI_PROVIDER_TESTING_GUIDE.md).