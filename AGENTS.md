# AGENTS.md - Neuro-Tasting EEG Project Development Guide

This file provides essential information for AI agents working on the Neuro-Tasting EEG data acquisition and visualization system.

## Project Overview

This is a Python-based desktop application that:
- Acquires EEG data via Bluetooth from "RFstar_7FEA" devices
- Performs real-time signal processing and neural metrics calculation
- Visualizes brain activity with cyberpunk-styled UI
- Implements a state machine architecture for different scenes (Menu, Session, History)
- Packages as a standalone executable using PyInstaller

**Technology Stack**: Python 3.11+, Pygame, NumPy, SciPy, Bleak (Bluetooth), PyInstaller

## Build & Development Commands

### Running the Application
```bash
# Main entry point
python main_app.py

# Alternative entry points (legacy)
python main_v16.py
```

### Building Executable
```bash
# Build using PyInstaller with the provided spec
pyinstaller main_app.spec

# Result: dist/main_app.exe
```

### Virtual Environment
- Python 3.11.7 (Anaconda)
- Virtual environment located in `./venv/`
- Activate when needed: `venv\Scripts\activate`

### Testing
- **No formal test suite currently exists**
- Manual testing by running the application
- Test Bluetooth connectivity with actual hardware
- Verify data recording to CSV files

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import os
import sys
import time
import threading

# Third-party imports second  
import pygame
import numpy as np
from scipy import signal
from bleak import BleakClient

# Local imports last - use absolute imports
from config import *
from bluetooth_core import EEGBluetoothReceiver
from ui_components import Panel, Button, TextInput
```

### Naming Conventions
- **Classes**: PascalCase (`GameApp`, `EEGAnalyzer`, `SceneMenu`)
- **Functions/Variables**: snake_case (`process_data`, `current_quality`)
- **Constants**: UPPER_SNAKE_CASE (`DESIGN_WIDTH`, `COLOR_BG`, `FPS`)
- **Private members**: prefix with underscore (`_compute_metrics_logic`)

### File Structure Patterns
- **Scene files**: `scene_*.py` (`scene_menu.py`, `scene_session.py`, `scene_history.py`)
- **Core modules**: descriptive names (`eeg_engine.py`, `bluetooth_core.py`, `ui_components.py`)
- **Configuration**: centralized in `config.py`

## Architecture Patterns

### State Machine Design
```python
class GameApp:
    def __init__(self):
        self.states = {}
        self.current_state_name = None
        self.current_scene = None
        
    def change_state(self, state_name, data=None):
        # Cleanup current scene
        if self.current_scene and hasattr(self.current_scene, 'exit_scene'):
            self.current_scene.exit_scene()
            
        # Transition to new scene
        if state_name == "MENU":
            self.current_scene = SceneMenu(self)
        elif state_name == "SESSION":
            self.current_scene = SceneSession(self, data)
```

### Threading Pattern
```python
class AsyncProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.daemon = True
        
    def run(self):
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.process(data)
                self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception:
                pass  # Graceful error handling
```

## Error Handling Guidelines

### Bluetooth/Communication
```python
try:
    from bluetooth_core import EEGBluetoothReceiver
    BLUETOOTH_AVAILABLE = True
except ImportError:
    print("Warning: 'bluetooth_core.py' not found. Bluetooth mode unavailable.")
    BLUETOOTH_AVAILABLE = False
```

### Queue Operations
```python
try:
    data = self.receiver_queue.get(timeout=0.1)
    # Process data
except queue.Empty:
    continue  # Expected timeout
except Exception:
    pass  # Graceful degradation
```

### Resource Management
```python
# Always check file existence
if not os.path.exists(data_path):
    try:
        os.makedirs(data_path)
    except Exception as e:
        print(f"Error creating dir: {e}")
        return fallback_path
```

## UI/UX Patterns

### Panel-Based Design
```python
# All UI elements should be contained in panels
class Panel:
    def __init__(self, x, y, w, h, title=None):
        self.rect = pygame.Rect(x, y, w, h)
        
    def draw(self, surface):
        # Semi-transparent background
        s = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        s.fill(COLOR_PANEL_BG)
        
        # Border decoration with corner highlights
        # Consistent cyberpunk styling
```

### Event Handling
```python
def handle_events(self, events):
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.change_state("MENU")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.btn_start.rect.collidepoint(event.pos):
                self.on_start_click()
```

## Data Processing Patterns

### EEG Signal Processing
```python
def process(self, new_chunk, reliability_mask=None):
    # 1. Filter data
    clean_ch1, self.zi_ch1 = signal.sosfilt(self.sos, new_chunk[0], zi=self.zi_ch1)
    
    # 2. Update circular buffer
    self.raw_buffer = np.roll(self.raw_buffer, -n_points, axis=1)
    self.raw_buffer[:, -n_points:] = clean_chunk
    
    # 3. Calculate metrics (reduced frequency)
    if self.frame_counter % self.calc_interval_frames == 0:
        self._compute_metrics_logic()
```

### EMA Smoothing
```python
def update_ema(key, new_value, alpha=0.15):
    old_value = self.metrics_ema.get(key, new_value)
    self.metrics_ema[key] = old_value + alpha * (new_value - old_value)
```

## Configuration Management

### Color Scheme (Cyberpunk Palette)
```python
# Core colors defined in config.py
COLOR_BG = (10, 10, 15)                    # Deep blue-black
COLOR_PANEL_BG = (16, 20, 25, 200)          # Semi-transparent black
COLOR_TEXT = (200, 220, 255)                # Light blue-white
COLOR_GOLD = (255, 215, 0)                  # Neon gold
COLOR_CYAN = (0, 255, 255)                  # Neon cyan
```

### Cross-Platform Paths
```python
def get_user_data_dir():
    """Gets cross-platform user data directory (Documents/NeuroTasting_Data)"""
    if platform.system() == "Windows":
        base_path = os.path.join(os.path.expanduser("~"), "Documents")
    else:
        base_path = os.path.join(os.path.expanduser("~"), "Documents")
    
    data_path = os.path.join(base_path, "NeuroTasting_Data")
    return data_path
```

## Development Best Practices

### 1. Resource Loading
```python
# Use the resource_path helper for PyInstaller compatibility
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
```

### 2. Frame Rate Management
```python
# Consistent 30 FPS for main application
self.clock.tick(30)

# 60 FPS for visualizations only
self.clock.tick(60)
```

### 3. Coordinate System
```python
# Design resolution: 2752x1356 (defined in config.py)
# All UI positioning uses design coordinates
# Automatic scaling handled in main_app.py
```

## Important Notes for Agents

### Working with This Codebase
1. **Always use `from config import *`** in scene and UI files
2. **Follow the state machine pattern** for new scenes
3. **Use the Panel class** for all UI containers
4. **Handle Bluetooth gracefully** - it may not be available
5. **Use queues for thread communication** between data acquisition and UI
6. **Test with actual hardware when possible** - simulated data differs from real EEG

### Common Pitfalls to Avoid
- Don't hardcode screen coordinates - use design resolution
- Don't block the main thread - use threading for heavy processing
- Don't ignore the EMA smoothing patterns in metrics calculations
- Don't skip the reliability mask handling in Bluetooth data
- Don't create UI elements without Panel containers

### Data Flow
```
Bluetooth Device → BluetoothCore → Queue → AsyncProcessor → EEGAnalyzer → Queue → Scene → UI
```

This architecture ensures real-time performance and clean separation of concerns.