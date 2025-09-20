# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-component emulation and AI gaming system with three main parts:

1. **PyBoy** - Game Boy emulator written in Python/Cython
2. **PyGBA** - Game Boy Advance emulator wrapper around mGBA
3. **AI Game Server** - Unified backend combining emulators with AI APIs

## Development Commands

### PyBoy (Primary Component)

**Building:**
```bash
cd PyBoy
make build                    # Build both ROMs and PyBoy
make build_pyboy             # Build only PyBoy
make build_rom               # Build only ROMs
```

**Testing:**
```bash
make test                    # Run all tests (Cython and PyPy)
make test_cython            # Run Cython tests only
make test_cpython_doctest   # Run doctest and internal tests
make benchmark              # Run performance benchmarks
```

**Development:**
```bash
make install                # Install after building
make clean                  # Clean build artifacts
```

**Individual test execution:**
```bash
pytest tests/test_basics.py -v
pytest tests/ --tb=short
```

**Package Management:**
```bash
pip install -e PyBoy/       # Install in development mode
```

### PyGBA Component

```bash
cd pygba
pip install -e .             # Install in development mode
```

### AI Game Server

```bash
cd ai-game-server
pip install -r requirements.txt
python server.py            # Start the server
```

## Architecture Overview

### PyBoy Architecture
- **Core** (`pyboy/core/`): CPU, LCD, memory management, cartridge handling
- **API** (`pyboy/api/`): External interface for bots/AI, screen access, memory scanning
- **Plugins** (`pyboy/plugins/`): Windowing systems, debug tools, game wrappers
- **Cartridge** (`pyboy/core/cartridge/`): Memory bank controllers (MBC1, MBC3, MBC5, etc.)

**Key Performance Note:** PyBoy uses Cython for performance-critical components. All `.py` files in `pyboy/` directory with corresponding `.pxd` files are compiled to C.

### Plugin System
PyBoy uses a plugin architecture for:
- Windowing (SDL2, OpenGL, Null)
- Debugging (breakpoints, memory inspection)
- Game-specific wrappers (Pokemon, Mario, Tetris)
- Recording/replay functionality

### AI Integration
The `ai-game-server` provides:
- RESTful API for emulator control
- Multi-AI API integration (Gemini, OpenRouter, NVIDIA NIM)
- State management for save/load functionality
- Action history tracking

## Code Style and Configuration

### Linting and Formatting
- Uses `ruff` for linting and formatting
- Line length: 120 characters
- Double quotes for strings
- Pre-commit hooks for automatic formatting

### Testing
- Uses `pytest` with parallel execution (`-n auto`)
- Test files in `tests/` directory
- Benchmark tests in `tests/test_benchmark.py`
- Requires ROM files for some tests (automatically downloaded)

### Performance Considerations
- Frame skipping for AI training: `pyboy.tick(15)` skips 15 frames
- Disable rendering for maximum performance: `pyboy.tick(target, False)`
- Cython compilation with aggressive optimizations

## Development Workflow

1. **Making Changes:** Edit Python files in `pyboy/` directory
2. **Building:** Run `make build_pyboy` to compile Cython extensions
3. **Testing:** Run `make test_cython` to verify changes
4. **API Changes:** Test with `make test_cpython_doctest`

## File Structure Notes

- **Boot ROMs:** Built from source in `extras/bootrom/`
- **Default ROMs:** Built in `extras/default_rom/`
- **Examples:** AI/bot examples in `extras/examples/`
- **Game Wrappers:** Pre-built AI interfaces for specific games

## Dependencies

### Core Dependencies
- `numpy`: Array operations and performance
- `pysdl2`: Display and input handling
- `pysdl2-dll`: SDL2 binaries
- `cython`: C compilation for performance

### Development Dependencies
- `pytest`: Testing framework
- `pytest-benchmark`: Performance testing
- `pytest-xdist`: Parallel test execution
- `pillow`: Image processing for screenshots
- `pyopengl`: Alternative rendering backend

## Multi-Component Integration

When working across components:
1. PyBoy provides the Game Boy emulation foundation
2. PyGBA extends to Game Boy Advance games
3. AI Game Server unifies both with AI API integration
4. Game wrappers provide easy AI interfaces for specific games