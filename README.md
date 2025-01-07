# AI-Clip-Generator
AI clip genenerator
# 🎬 Automatic Video Subtitler

Automatically add subtitles to your videos using speech recognition! This tool transcribes speech from your videos and overlays the text as beautiful, customizable subtitles.

## ✨ Features

- 🎯 Automatic speech recognition using Google's Speech Recognition API
- 🎨 Customizable subtitle appearance (font, size, color, outline)
- ⚡ Smart text wrapping and positioning
- 🔄 Timestamp synchronization
- 📱 Support for multiple video formats

## 🚀 Installation

1. First, make sure you have Python 3.7+ installed on your system.

2. Install the required packages:
```bash
pip install moviepy SpeechRecognition Pillow numpy
```

3. Additional system requirements:
   - FFmpeg (required by moviepy)
   - A working internet connection (for Google Speech Recognition)

## 💻 Usage

Basic usage:
```bash
python script.py video_path
```

Example with custom settings:
```bash
python script.py my_video.mp4 --font-size 40 --font-color yellow --outline-width 3
```

### 🎮 Available Options

- `video_path`: Path to your input video file
- `--output`: Custom output path (optional)
- `--font-path`: Path to TTF font file (default: Arial)
- `--font-size`: Font size in pixels (default: 30)
- `--font-color`: Text color (hex code or name) (default: white)
- `--outline-color`: Outline color (hex code or name) (default: black)
- `--outline-width`: Width of text outline in pixels (default: 2)
- `--line-spacing`: Spacing between lines in pixels (default: 4)
- `--bottom-padding`: Padding from bottom of screen in pixels (default: 50)
- `--width-percent`: Width of text box as percentage of video width (0.0-1.0) (default: 0.8)

### 🎨 Supported Colors

You can use either:
- Color names: `red`, `white`, `black`, `yellow`, `blue`, `green`
- Hex codes: `#FF0000`, `#FFFFFF`, etc.

## 📝 Example Commands

1. Basic usage with default settings:
```bash
python script.py myvideo.mp4
```

2. Custom font and colors:
```bash
python script.py myvideo.mp4 --font-size 45 --font-color yellow --outline-color black --outline-width 3
```

3. Custom output path:
```bash
python script.py input.mp4 --output subtitled_video.mp4
```

4. Adjust subtitle positioning:
```bash
python script.py video.mp4 --bottom-padding 70 --width-percent 0.7
```

## ⚠️ Important Notes

- The script requires an internet connection for speech recognition
- Processing time depends on video length and system performance
- Font paths may need adjustment based on your operating system
- Make sure you have sufficient disk space for temporary files

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.