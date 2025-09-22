# AI Voice Assistant Speed & Cost Comparison Tool

This tool compares the performance and cost of three AI voice assistant approaches:

1. **OpenAI Realtime API (default)**: Direct audio-to-audio processing using the default realtime model
2. **OpenAI Realtime API (GPT-4o)**: Direct audio-to-audio processing using GPT-4o Realtime
3. **Cartesia Pipeline**: Speech-to-Text (Cartesia) → GPT-4o (OpenAI) → Text-to-Speech (Cartesia)

## Features

- Records audio from your default microphone (5 seconds)
- Sends the same audio to all three processing approaches
- Measures response time for each approach (excluding playback time)
- Calculates estimated costs based on API pricing
- Displays text transcripts of both input and output
- Plays back all three responses sequentially for comparison
- Provides detailed timing and cost analysis

## Prerequisites

### System Requirements

- Python 3.8 or higher
- macOS, Linux, or Windows
- Working microphone
- Audio output device (speakers/headphones)

### macOS Additional Setup

Install PortAudio (required for audio recording):
```bash
brew install portaudio
```

## Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export CARTESIA_API_KEY="your-cartesia-api-key"
```

Alternatively, create a `.env` file in the project directory:
```
OPENAI_API_KEY=your-openai-api-key
CARTESIA_API_KEY=your-cartesia-api-key
```

## Usage

Run the comparison script:
```bash
python audio_model_comparison.py
```

The script will:
1. Prompt you to record a 5-second audio message
2. Test OpenAI Realtime API (default model) and play the response
3. Test OpenAI Realtime API (GPT-4o) and play the response  
4. Test Cartesia + GPT-4o pipeline and play the response
5. Show text transcripts for all approaches (input + output)
6. Display detailed timing and cost analysis for each approach
7. Provide speed and cost rankings comparing all three methods

## Understanding the Results

### Response Time
- **OpenAI Realtime**: Total time from sending audio to receiving response (playback not included)
- **Cartesia Pipeline**: Broken down into STT, GPT-4o, and TTS processing times (playback not included)

### Text Transcripts
- **Input Transcript**: Shows what the system understood from your speech
- **Output Transcript**: Shows the text response before it's converted to speech
- Available for both OpenAI Realtime and Cartesia pipeline approaches

### Cost Calculation

#### OpenAI Realtime API (Default Model)
- Input: $32 per 1M tokens
- Output: $64 per 1M tokens

#### OpenAI Realtime API (GPT-4o)
- Input: $40 per 1M tokens  
- Output: $80 per 1M tokens

#### Cartesia Pipeline (Pro tier pricing: $5/month for 100K credits)
- STT (Ink-Whisper): 1 credit/second = $0.00005/second
- GPT-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
- TTS (Sonic): 1 credit/character = $0.00005/character

**Note**: Token counts are calculated from actual transcripts for accurate cost comparison.

## Output Files

The script creates several files:
- `input_audio.wav`: Your recorded message
- `openai_realtime_default_response.wav`: Response from OpenAI Realtime API (default model)
- `openai_realtime_mini_response.wav`: Response from OpenAI Realtime API (GPT-4o)
- `cartesia_pipeline_response.wav`: Response from Cartesia pipeline

## Troubleshooting

### "No audio devices found"
- Ensure your microphone is connected and recognized by your system
- On macOS, check System Preferences → Security & Privacy → Microphone

### "API key not found"
- Verify that your environment variables are set correctly
- Ensure there are no extra spaces in your API keys

### WebSocket connection errors
- Check your internet connection
- Verify that your OpenAI API key has access to the Realtime API
- The Realtime API may have regional restrictions

## Technical Details

- **Audio Format**: 24kHz mono PCM16 for compatibility with both APIs
- **OpenAI Realtime**: Uses WebSocket with base64-encoded audio chunks and `gpt-4o-realtime-preview-2024-10-01`
- **Cartesia Integration**: Uses direct HTTP API calls (no SDK dependency)
- **Cartesia Voice**: ID `694f9389-aac1-45b6-b726-9d9369183238` with `sonic-2` model
- **Cartesia Pricing**: Pro tier (1 credit = $0.00005), STT = 1 credit/second, TTS = 1 credit/character
- **Timing**: Response time measurement stops when audio is received, not including playback
- **Timeouts**: 30-second timeout for API responses
- **Audio Generation**: ~750-800 characters of text generates ~1 minute of audio

## Example Output

Here are some real test results showing the tool in action:

```
🎯 Audio Model Comparison Test
============================================================

🎤 Recording for 5 seconds...
Please speak your message now...
✅ Recording complete!

🚀 Starting tests...

📡 Testing OpenAI Realtime API (default model)...
📊 Input audio: 24000Hz, 5.00s, 240000 bytes
📊 PCM data ready: 240000 bytes
🔄 Connecting to OpenAI Realtime API...
✅ Connected to OpenAI Realtime API
📋 Session configuration sent
📤 Sending audio input...
📊 Audio peak level: 1496 (out of 32767)
📤 Sent audio data: 240000 bytes as 320000 base64 chars
📤 Committed audio buffer
✅ Audio buffer committed
📤 Requested response generation
⏳ Waiting for response...
🎤 Input transcript: What is the capital of Germany?
🎵 Received audio chunk: 4800 bytes (total: 4800 bytes)
...
💬 Output transcript: The capital of Germany is Berlin. It's a vibrant city known for its history, culture, and modern landmarks.
✅ Received audio response (336000 bytes)
📊 Input tokens (from transcript): 7
   Output tokens (from transcript): 26
⏱️  Total response time: 4.39s (playback not included)

🔊 Playing OpenAI Realtime (default) response...
✅ OpenAI Realtime default response completed

============================================================
📡 Testing OpenAI Realtime API (GPT-4o)...
============================================================
🎤 Input transcript: What is the capital of Germany?
💬 Output transcript: Die Hauptstadt von Deutschland ist Berlin.
📊 Input tokens (from transcript): 7
   Output tokens (from transcript): 10
⏱️  Total response time: 3.48s (playback not included)

🔊 Playing OpenAI Realtime GPT-4o response...
✅ OpenAI Realtime GPT-4o response completed

============================================================
🔄 Now testing Cartesia + GPT-4o Pipeline...
============================================================
📝 Transcript: What is the capital of Germany?
🤖 GPT-4o response: The capital of Germany is Berlin.
🔧 Using Cartesia direct API for TTS...
📊 Cartesia TTS response: 111454 bytes
   Expected duration: 2.32 seconds
⏱️  Total pipeline time: 4.46s (playback not included)

🔊 Playing Cartesia + GPT-4o response...
✅ Cartesia pipeline response completed

============================================================
📊 OpenAI Realtime API (default) Results
============================================================
⏱️  Response Time: 4.39 seconds
🤖 Model: gpt-realtime
📥 Input Tokens: 7
📤 Output Tokens: 26
💰 Input Cost: $0.0002
💰 Output Cost: $0.0017
💰 Total Cost: $0.0019

🎤 What you said: "What is the capital of Germany?"
💬 AI response: "The capital of Germany is Berlin. It's a vibrant city known for its history, culture, and modern landmarks."

============================================================
📊 OpenAI Realtime API (GPT-4o) Results
============================================================
⏱️  Response Time: 3.48 seconds
🤖 Model: gpt-4o-realtime-preview
📥 Input Tokens: 7
📤 Output Tokens: 10
💰 Input Cost: $0.0003
💰 Output Cost: $0.0008
💰 Total Cost: $0.0011

🎤 What you said: "What is the capital of Germany?"
💬 AI response: "Die Hauptstadt von Deutschland ist Berlin."

============================================================
📊 Cartesia + GPT-4o Pipeline Results
============================================================
⏱️  Response Time: 4.46 seconds

⏱️  Breakdown:
   - Speech-to-Text: 1.56s
   - GPT-4 Processing: 0.92s
   - Text-to-Speech: 1.98s

💰 Cost Breakdown:
   - STT Cost: $0.0003
   - LLM Cost: $0.0001
     • Input tokens: 7
     • Output tokens: 7
   - TTS Cost: $0.0016
   - Total Cost: $0.0020

🎤 What you said: "What is the capital of Germany?"
💬 AI response: "The capital of Germany is Berlin."

============================================================
📈 COMPARISON SUMMARY
============================================================

⏱️  Speed Ranking (fastest to slowest):
   1. OpenAI Realtime (GPT-4o): 3.48s
   2. OpenAI Realtime (default): 4.39s
   3. Cartesia + GPT-4o: 4.46s

💰 Cost Ranking (cheapest to most expensive):
   1. OpenAI Realtime (GPT-4o): $0.0011
   2. OpenAI Realtime (default): $0.0019
   3. Cartesia + GPT-4o: $0.0020

============================================================

### Complex Question Example:
🎤 Input transcript: Is a tomato a fruit or a vegetable?

OpenAI Realtime (default) Response:
💬 "Botanisch gesehen ist eine Tomate eine Frucht, weil sie aus der Blüte der Pflanze hervorgeht und Samen enthält. Aber in der Küche wird sie meist als Gemüse verwendet, weil sie eher herzhaft ist. Also: wissenschaftlich ist es eine Frucht, kulinarisch wird sie oft als Gemüse betrachtet."
⏱️  Response Time: 6.18s
💰 Total Cost: $0.0048

OpenAI Realtime (GPT-4o) Response:
💬 "Botanisch gesehen ist eine Tomate eine Frucht, weil sie aus der Blüte der Pflanze entsteht und Samen enthält. Aber in der Küche wird sie oft als Gemüse betrachtet, weil sie herzhaft zubereitet wird. Es kommt also darauf an, ob man die botanische oder die kulinarische Perspektive betrachtet."
⏱️  Response Time: 6.42s
💰 Total Cost: $0.0061

Cartesia + GPT-4o Response:
💬 "A tomato is botanically a fruit because it develops from the ovary of a flower and contains seeds. However, in culinary terms, it's often treated as a vegetable because of its savory flavor."
⏱️  Response Time: 11.86s
💰 Total Cost: $0.0102

⏱️  Speed Ranking (fastest to slowest):
   1. OpenAI Realtime (default): 6.18s
   2. OpenAI Realtime (GPT-4o): 6.42s
   3. Cartesia + GPT-4o: 11.86s

💰 Cost Ranking (cheapest to most expensive):
   1. OpenAI Realtime (default): $0.0048
   2. OpenAI Realtime (GPT-4o): $0.0061
   3. Cartesia + GPT-4o: $0.0102

============================================================

### Simple Question Example:
🎤 Input transcript: How many legs does a spider have?

OpenAI Realtime (default) Response:
💬 "A spider has eight legs. These legs are jointed and help it move around, hunt, and build webs."
⏱️  Response Time: 6.55s
💰 Total Cost: $0.0017

OpenAI Realtime (GPT-4o) Response:
💬 "A spider has eight legs."
⏱️  Response Time: 3.31s
💰 Total Cost: $0.0008

Cartesia + GPT-4o Response:
💬 "A spider has eight legs."
⏱️  Response Time: 3.07s
💰 Total Cost: $0.0015

⏱️  Speed Ranking (fastest to slowest):
   1. Cartesia + GPT-4o: 3.07s
   2. OpenAI Realtime (GPT-4o): 3.31s
   3. OpenAI Realtime (default): 6.55s

💰 Cost Ranking (cheapest to most expensive):
   1. OpenAI Realtime (GPT-4o): $0.0008
   2. Cartesia + GPT-4o: $0.0015
   3. OpenAI Realtime (default): $0.0017

============================================================

### Another Simple Question Example:
🎤 Input transcript: Name one animal that can fly.

OpenAI Realtime (default) Response:
💬 "Sure! A bird, like an eagle, can fly."
⏱️  Response Time: 6.12s
💰 Total Cost: $0.0008

OpenAI Realtime (GPT-4o) Response:
💬 "Ein Tier, das fliegen kann, ist der Vogel, zum Beispiel ein Adler. Vögel sind bekannt dafür, dass sie fliegen können."
⏱️  Response Time: 5.00s
💰 Total Cost: $0.0026

Cartesia + GPT-4o Response:
💬 "A bird, like a sparrow, can fly."
⏱️  Response Time: 4.42s
💰 Total Cost: $0.0020

⏱️  Speed Ranking (fastest to slowest):
   1. Cartesia + GPT-4o: 4.42s
   2. OpenAI Realtime (GPT-4o): 5.00s
   3. OpenAI Realtime (default): 6.12s

💰 Cost Ranking (cheapest to most expensive):
   1. OpenAI Realtime (default): $0.0008
   2. Cartesia + GPT-4o: $0.0020
   3. OpenAI Realtime (GPT-4o): $0.0026

============================================================

### Yes/No Question Example (with short instruction):
🎤 Input transcript: Is ice colder than water, yes or no?

OpenAI Realtime (default) Response:
💬 "Yes, ice is colder than liquid water."
⏱️  Response Time: 3.39s
💰 Total Cost: $0.0009

OpenAI Realtime (GPT-4o) Response:
💬 "Yes."
⏱️  Response Time: 2.93s
💰 Total Cost: $0.0201

Cartesia + GPT-4o Response:
💬 "Yes."
⏱️  Response Time: 2.60s
💰 Total Cost: $0.0005

⏱️  Speed Ranking (fastest to slowest):
   1. Cartesia + GPT-4o: 2.60s
   2. OpenAI Realtime (GPT-4o): 2.93s
   3. OpenAI Realtime (default): 3.39s

💰 Cost Ranking (cheapest to most expensive):
   1. Cartesia + GPT-4o: $0.0005
   2. OpenAI Realtime (default): $0.0009
   3. OpenAI Realtime (GPT-4o): $0.0201

Note: GPT-4o shows higher cost due to input token counting issue (500 tokens instead of 9)
```

## About This Tool

This tool provides a comprehensive comparison of three different approaches to building AI voice assistants. It tests the same audio input across all three methods and provides detailed analysis of:

- **Response Speed**: How quickly each approach processes and responds to your voice input
- **Cost Analysis**: Accurate token-based cost calculations for each approach  
- **Quality Comparison**: Text transcripts showing what each system understood and responded
- **Technical Insights**: Detailed breakdowns of processing times and token usage

**Key Features:**
- Uses actual transcripts for accurate token counting (not estimates)
- Fair comparison by excluding system prompt tokens
- Real-time audio processing and playback
- Comprehensive speed and cost rankings

## Customization

You can modify the following parameters in the script:
- `duration`: Recording duration (default: 5 seconds)
- `sample_rate`: Audio sample rate (default: 24000 Hz)
- Response timeout values (default: 30 seconds)
- GPT-4o model parameters (temperature, max_tokens)
- Cartesia voice ID and model selection
