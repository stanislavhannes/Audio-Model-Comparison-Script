# AI Voice Assistant Speed & Cost Comparison Tool

This tool compares the performance and cost of two AI voice assistant approaches:

1. **OpenAI Realtime API**: Direct audio-to-audio processing using GPT-4o Realtime
2. **Cartesia Pipeline**: Speech-to-Text (Cartesia) → GPT-4o (OpenAI) → Text-to-Speech (Cartesia)

## Features

- Records audio from your default microphone (5 seconds)
- Sends the same audio to both processing pipelines
- Measures response time for each approach (excluding playback time)
- Calculates estimated costs based on API pricing
- Displays text transcripts of both input and output
- Plays back both responses sequentially for comparison
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
2. Process the audio through OpenAI Realtime API
3. Show text transcripts (what you said + AI response)
4. Play the OpenAI response immediately when received
5. Process the same audio through Cartesia + GPT-4o pipeline
6. Show text transcripts for the pipeline
7. Play the Cartesia response when ready
8. Display detailed timing and cost analysis for both approaches

## Understanding the Results

### Response Time
- **OpenAI Realtime**: Total time from sending audio to receiving response (playback not included)
- **Cartesia Pipeline**: Broken down into STT, GPT-4o, and TTS processing times (playback not included)

### Text Transcripts
- **Input Transcript**: Shows what the system understood from your speech
- **Output Transcript**: Shows the text response before it's converted to speech
- Available for both OpenAI Realtime and Cartesia pipeline approaches

### Cost Calculation

#### OpenAI Realtime API
- Input: $32 per 1M audio tokens (~$0.06 per minute)
- Output: $64 per 1M audio tokens (~$0.24 per minute)

#### Cartesia Pipeline (Pro tier pricing: $5/month for 100K credits)
- STT (Ink-Whisper): 1 credit/second = $0.00005/second (~$0.18/hour)
- GPT-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
- TTS (Sonic): 1 credit/character = $0.00005/character (~$0.05 per 1000 chars)

## Output Files

The script creates several files:
- `input_audio.wav`: Your recorded message
- `openai_realtime_response.wav`: Response from OpenAI Realtime API
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

📡 Testing OpenAI Realtime API...
📊 Input audio: 24000Hz, 5.00s, 240000 bytes
📊 PCM data ready: 240000 bytes
🔄 Connecting to OpenAI Realtime API...
✅ Connected to OpenAI Realtime API
📋 Session configuration sent
📤 Sending audio input...
📊 Audio peak level: 3491 (out of 32767)
📤 Sent audio data: 240000 bytes as 320000 base64 chars
📤 Committed audio buffer
✅ Audio buffer committed
📤 Requested response generation
⏳ Waiting for response...
🎤 Input transcript: What is the capital of Germany?
🎵 Received audio chunk: 4800 bytes (total: 4800 bytes)
...
💬 Output transcript: The capital of Germany is Berlin.
✅ Received audio response (124800 bytes)
   Expected duration: 2.60 seconds
⏱️  Total response time: 3.15s (playback not included)

============================================================
📊 OpenAI Realtime API Results
============================================================
⏱️  Response Time: 3.15 seconds
📥 Input Tokens: 500
📤 Output Tokens: 260
💰 Input Cost: $0.0160
💰 Output Cost: $0.0166
💰 Total Cost: $0.0326

🎤 What you said: "What is the capital of Germany?"
💬 AI response: "The capital of Germany is Berlin."

============================================================
📊 Cartesia + GPT-4 Pipeline Results
============================================================
⏱️  Response Time: 2.71 seconds

⏱️  Breakdown:
   - Speech-to-Text: 0.95s
   - GPT-4 Processing: 0.58s
   - Text-to-Speech: 1.17s

💰 Cost Breakdown:
   - STT Cost: $0.0003
   - LLM Cost: $0.0002
     • Input tokens: 33
     • Output tokens: 7
   - TTS Cost: $0.0016
   - Total Cost: $0.0021

🎤 What you said: "What is the capital of Germany?"
💬 AI response: "The capital of Germany is Berlin."

============================================================
📈 COMPARISON SUMMARY
============================================================

⏱️  Speed: Cartesia Pipeline was 1.2x faster
   - OpenAI Realtime: 3.15s
   - Cartesia Pipeline: 2.71s
   - Difference: 0.44s

💰 Cost: Cartesia Pipeline was 15.9x cheaper
   - OpenAI Realtime: $0.0326
   - Cartesia Pipeline: $0.0021
   - Difference: $0.0306

============================================================

### Test 2: Accidently Multilingual Response
🎤 Input transcript: Who was born on Christmas?
💬 Output transcript: Traditionell wird angenommen, dass Jesus Christus an Weihnachten geboren wurde. Das ist der Grund, warum viele Menschen am 25. Dezember seinen Geburtstag feiern. Es ist also eine religiöse und kulturelle Überlieferung, die mit dem Weihnachtsfest verbunden ist.
⏱️  Total response time: 6.53s (playback not included)

📊 Cartesia + GPT-4 Pipeline Results
⏱️  Response Time: 3.69 seconds
📝 Transcript: Who was born on Christmas?
🤖 GPT-4o response: Traditionally, Christmas is celebrated as the birth of Jesus Christ.

💰 OpenAI Realtime: $0.1392
💰 Cartesia Pipeline: $0.0039

⏱️  Speed: Cartesia Pipeline was 1.8x faster
   - OpenAI Realtime: 6.53s
   - Cartesia Pipeline: 3.69s
   - Difference: 2.85s

💰 Cost: Cartesia Pipeline was 36.1x cheaper
   - OpenAI Realtime: $0.1392
   - Cartesia Pipeline: $0.0039
   - Difference: $0.1353

============================================================

### Test 3: Complex Question
🎤 Input transcript: Is a tomato a vegetable or a fruit?
💬 Output transcript: Great question! A tomato is actually both. Botanically, it's a fruit because it develops from the ovary of a flower and contains seeds. But in culinary terms, it's treated as a vegetable because of its savory flavor. So, in the kitchen, most people call it a vegetable, even though scientifically it's a fruit.
⏱️  Total response time: 5.93s (playback not included)

📊 Cartesia + GPT-4 Pipeline Results
⏱️  Response Time: 7.91 seconds
📝 Transcript: Is a tomato a vegetable or a fruit?
🤖 GPT-4o response: Botanically, a tomato is a fruit because it develops from the ovary of a flower and contains seeds. However, in culinary terms, it is often treated as a vegetable due to its savory flavor.

💰 OpenAI Realtime: $0.1267
💰 Cartesia Pipeline: $0.0102

⏱️  Speed: OpenAI Realtime was 1.3x faster
   - OpenAI Realtime: 5.93s
   - Cartesia Pipeline: 7.91s
   - Difference: 1.98s

💰 Cost: Cartesia Pipeline was 12.5x cheaper
   - OpenAI Realtime: $0.1267
   - Cartesia Pipeline: $0.0102
   - Difference: $0.1166
```

## Summary

Based on testing, the **Cartesia pipeline shows interesting speed patterns and is significantly cheaper** than OpenAI's Realtime API:

- **Speed**: Performance depends on the underlying LLM model
  - **Cartesia + GPT-4o**: Often faster due to lighter GPT-4o model
  - **OpenAI Realtime (GPT-5)**: Slower due to more powerful but heavier model
  - **OpenAI Realtime (GPT-4o-mini)**: Consistently faster than Cartesia pipeline
  - Overall differences typically under 3 seconds
- **Cost**: Cartesia pipeline is **12-36x cheaper** than OpenAI Realtime
- **Model difference**: OpenAI Realtime likely uses GPT-5 (as of Sept 2025), while the pipeline uses GPT-4o

**Potential improvements for Cartesia pipeline:**
- Could be faster using [Cartesia WebSocket STT/TTS](https://docs.cartesia.ai/2024-11-13/api-reference/stt/stt) instead of HTTP API
- Extensive voice customization and personalization options available
- Flexible model selection - easily swap GPT-4o for other models

**Cost alternatives:**
- GPT-4o-mini Realtime would cost 1/3 of current Realtime pricing, but still significantly more expensive than Cartesia pipeline
- Cartesia + GPT-4o offers better quality than GPT-4o-mini at much lower cost

## Customization

You can modify the following parameters in the script:
- `duration`: Recording duration (default: 5 seconds)
- `sample_rate`: Audio sample rate (default: 24000 Hz)
- Response timeout values (default: 30 seconds)
- GPT-4o model parameters (temperature, max_tokens)
- Cartesia voice ID and model selection
