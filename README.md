# Prism - AI-Powered Screen Reader & Time Tracker

<div align="center">
  <h3>🔍 Intelligent Desktop Ecosystem That Understands Your Work</h3>
  <p><em>Analyzing and breaking down complex activities into understandable components</em></p>
</div>

## 🌟 Core Vision

An intelligent desktop ecosystem that understands your work patterns, learns from your behavior, and evolves into a comprehensive productivity assistant.

## 🚀 Current Features (Phase 1 - MVP)

### Observer Agent
- **Screen Analysis**: Periodic screenshots with OCR and window detection
- **Activity Classification**: ML-powered categorization of work activities
- **Privacy-First**: All processing happens locally on your machine
- **Basic Time Tracking**: Application usage and activity duration logging
- **Simple Dashboard**: Real-time activity monitoring and daily summaries

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 Prism Core                      │
├─────────────────────────────────────────────────┤
│  • Event Bus (Agent Communication)             │
│  • Privacy & Security Layer                    │
│  • Data Management (Local SQLite)              │
│  • Agent Registry & Plugin System              │
└─────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│              Observer Agent                     │
├─────────────────────────────────────────────────┤
│  • Screen Capture       • OCR Processing       │
│  • Window Detection     • Activity Classification│
│  • Time Tracking        • Data Storage         │
└─────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Python 3.9+
- **Screen Capture**: PyAutoGUI, Pillow
- **OCR**: Tesseract (via pytesseract)
- **ML Classification**: scikit-learn, transformers
- **Database**: SQLite
- **GUI**: Tkinter (future: PyQt or Electron)
- **Privacy**: Local-only processing, encrypted storage

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Tesseract OCR engine

### macOS Setup
```bash
# Install Tesseract
brew install tesseract

# Clone the repository
git clone <your-repo-url>
cd prism-ai-screen-reader

# Install Python dependencies
pip install -r requirements.txt

# Initialize the database
python -m prism.core.database init

# Run Prism
python -m prism.main
```

### Security Permissions (macOS)
Prism requires screen capture permissions. When first running:
1. Go to System Preferences → Security & Privacy → Privacy
2. Select "Screen Recording" from the left sidebar
3. Add Terminal (or your Python environment) to the allowed applications

## 🎯 Usage

```bash
# Start the Observer agent
python -m prism.main --mode observer

# View activity dashboard
python -m prism.dashboard

# Export activity data
python -m prism.export --format json --date-range 7d
```

## 🗺️ Roadmap

### ✅ Phase 1: Foundation (Current)
- [x] Screen capture and OCR
- [x] Basic activity classification
- [x] Local data storage
- [x] Simple time tracking
- [ ] Dashboard interface

### 🔄 Phase 2: Intelligence Layer (Next)
- [ ] Pattern recognition and work rhythm analysis
- [ ] Automatic project detection
- [ ] Productivity metrics and insights
- [ ] Smart notifications and break reminders

### 🔮 Phase 3: Automation & Assistance
- [ ] Workflow automation suggestions
- [ ] Context-aware assistance
- [ ] Resource recommendations
- [ ] Meeting intelligence

### 🌐 Phase 4: Ecosystem Expansion
- [ ] Specialized agent marketplace
- [ ] Team collaboration insights
- [ ] Health and wellness monitoring
- [ ] Advanced integrations

## 🔒 Privacy & Security

- **Local Processing**: All AI analysis happens on your device
- **Encrypted Storage**: Sensitive data is encrypted at rest
- **No Cloud Dependencies**: Works completely offline
- **Granular Controls**: Fine-tune what data is collected and analyzed
- **Data Ownership**: You own and control all your data

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](issues/)
- 💬 [Discussions](discussions/)

---

<div align="center">
  <p><em>Built with ❤️ for productivity enthusiasts who value privacy</em></p>
</div> 