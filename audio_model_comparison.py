#!/usr/bin/env python3
"""
Audio Model Comparison Script

This script compares two AI audio processing approaches:
1. OpenAI's GPT Realtime API (direct audio-to-audio)
2. Cartesia STT -> OpenAI GPT-4 -> Cartesia TTS pipeline

It measures response times and calculates costs for both approaches.
"""

import os
import json
import time
import wave
import struct
import threading
import queue
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import requests
import websocket
from datetime import datetime


class AudioRecorder:
    """Handles audio recording from the default microphone."""
    
    def __init__(self, duration: int = 5, sample_rate: int = 24000):
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = 1
    
    def record(self, filename: str) -> None:
        """Record audio from the default microphone and save to file."""
        print(f"\n🎤 Recording for {self.duration} seconds...")
        print("Please speak your message now...")
        
        # Record audio
        audio_data = sd.rec(
            int(self.sample_rate * self.duration),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        sd.wait()  # Wait for recording to finish
        
        print("✅ Recording complete!")
        
        # Save to WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())


class AudioPlayer:
    """Handles audio playback."""
    
    @staticmethod
    def play_file(filename: str, sample_rate: Optional[int] = None) -> None:
        """Play audio from a WAV file."""
        try:
            with wave.open(filename, 'rb') as wf:
                # Get file properties
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                actual_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / actual_rate
                
                # Read audio data
                audio_data = wf.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                print(f"🔊 Playing audio response:")
                print(f"   Sample rate: {actual_rate}Hz")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Frames: {n_frames}")
                print(f"   Array size: {len(audio_array)}")
                
                # Ensure we're not clipping the audio
                sd.play(audio_array, samplerate=actual_rate, blocking=True)
                # Add a small delay to ensure playback completes
                time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            import traceback
            traceback.print_exc()


class OpenAIRealtimeClient:
    """Client for OpenAI's Realtime API using WebSocket."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # OpenAI Realtime API WebSocket URL with model selection
        # Current: GPT-5 (most capable, slower, more expensive)
        # Alternative: Change to "gpt-4o-mini-realtime-preview-2024-10-01" for faster performance
        self.ws_url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        self.response_queue = queue.Queue()
        self.ws = None
        self.session_id = None
        self.audio_buffer = bytearray()
        self.response_complete = threading.Event()
        self.input_transcript = ""
        self.output_transcript = ""
        
    def process_audio(self, audio_file: str) -> Tuple[Optional[bytes], float, dict]:
        """Send audio to OpenAI Realtime API and get response."""
        start_time = time.time()
        
        # Read audio file and convert to 24kHz as required by API
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            audio_data = wf.readframes(frames)
            input_duration = frames / float(rate)
        
        print(f"📊 Input audio: {rate}Hz, {input_duration:.2f}s, {len(audio_data)} bytes")
        
        # Convert to 24kHz if needed (API requires 24kHz)
        if rate != 24000:
            print(f"⚠️  Converting audio from {rate}Hz to 24000Hz for Realtime API...")
            # Simple resampling - for production use scipy.signal.resample
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Resample to 24kHz
            resample_ratio = 24000 / rate
            new_length = int(len(audio_array) * resample_ratio)
            resampled = np.interp(
                np.linspace(0, len(audio_array) - 1, new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
            pcm_data = resampled.tobytes()
        else:
            # Use the raw audio data directly
            pcm_data = audio_data
        
        print(f"📊 PCM data ready: {len(pcm_data)} bytes")
        
        # Estimate input tokens (OpenAI Realtime API token estimation)
        # Note: This is an estimate - actual token usage may vary
        input_tokens = int(input_duration * 100)  # ~100 tokens per second
        
        response_audio = None
        output_tokens = 0
        error = None
        self.audio_buffer = bytearray()
        self.response_complete.clear()
        self.input_transcript = ""
        self.output_transcript = ""
        
        try:
            # Create WebSocket connection
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            print("🔄 Connecting to OpenAI Realtime API...")
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for session to be created
            session_timeout = 5
            session_start = time.time()
            while not self.session_id and time.time() - session_start < session_timeout:
                time.sleep(0.1)
            
            if not self.session_id:
                raise Exception("Failed to establish session with Realtime API")
            
            # Send audio input
            if self.ws.sock and self.ws.sock.connected:
                print("📤 Sending audio input...")
                
                # Convert audio to base64 (not hex!)
                import base64
                
                # Verify we have audio data
                if len(pcm_data) == 0:
                    raise Exception("No audio data to send!")
                
                # Check if audio is silent (all zeros or very low values)
                import numpy as np
                audio_check = np.frombuffer(pcm_data, dtype=np.int16)
                audio_max = np.max(np.abs(audio_check))
                print(f"📊 Audio peak level: {audio_max} (out of 32767)")
                
                audio_base64 = base64.b64encode(pcm_data).decode('utf-8')
                
                # First, append audio to buffer
                audio_append_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64  # Base64, not hex!
                }
                self.ws.send(json.dumps(audio_append_event))
                print(f"📤 Sent audio data: {len(pcm_data)} bytes as {len(audio_base64)} base64 chars")
                
                # Small delay to ensure audio is in buffer
                time.sleep(0.2)
                
                # Then commit the buffer
                commit_event = {
                    "type": "input_audio_buffer.commit"
                }
                self.ws.send(json.dumps(commit_event))
                print("📤 Committed audio buffer")
                
                # Wait a bit for server to process the audio
                time.sleep(0.5)
                
                # Explicitly request a response
                response_event = {
                    "type": "response.create"
                }
                self.ws.send(json.dumps(response_event))
                print("📤 Requested response generation")
                
                print("⏳ Waiting for response...")
                
                # Wait for response completion
                if self.response_complete.wait(timeout=30):
                    if len(self.audio_buffer) > 0:
                        response_audio = bytes(self.audio_buffer)
                        # Estimate output tokens and duration
                        output_duration = len(response_audio) / (24000 * 2)  # 24kHz, 16-bit
                        output_tokens = int(output_duration * 100)
                        print(f"✅ Received audio response ({len(response_audio)} bytes)")
                        print(f"   Expected duration: {output_duration:.2f} seconds")
                    else:
                        error = "No audio data received"
                else:
                    error = "Response timeout"
            
            # Close WebSocket with a small delay to ensure all data is received
            if self.ws:
                time.sleep(0.5)  # Small delay to ensure all audio chunks are received
                self.ws.close()
                ws_thread.join(timeout=2)
                
        except Exception as e:
            error = str(e)
            print(f"❌ OpenAI Realtime API error: {e}")
        
        # Stop timer here - before any playback
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"⏱️  Total response time: {elapsed_time:.2f}s (playback not included)")
        
        # Prepare cost info
        # WARNING: These are estimates based on available pricing info
        # Actual costs may vary - verify with current OpenAI pricing
        cost_info = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': (input_tokens / 1_000_000) * 32,  # $32 per 1M tokens (estimated)
            'output_cost': (output_tokens / 1_000_000) * 64,  # $64 per 1M tokens (estimated)
            'total_cost': 0,
            'error': error
        }
        cost_info['total_cost'] = cost_info['input_cost'] + cost_info['output_cost']
        cost_info['input_transcript'] = self.input_transcript
        cost_info['output_transcript'] = self.output_transcript
        
        return response_audio, elapsed_time, cost_info
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        print("✅ Connected to OpenAI Realtime API")
        
        # Send session.update to configure the session
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "You are a helpful AI assistant. Always respond with speech to the user's audio input. Keep responses brief and conversational.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": None,  # Disable auto turn detection to control manually
                "temperature": 0.8
            }
        }
        ws.send(json.dumps(session_update))
        print("📋 Session configuration sent")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            event_type = data.get('type')
            
            if event_type == 'session.created':
                self.session_id = data.get('session', {}).get('id')
                session_config = data.get('session', {})
                print(f"📋 Session created: {self.session_id}")
                print(f"   - Voice: {session_config.get('voice', 'N/A')}")
                print(f"   - Audio format: {session_config.get('input_audio_format', 'N/A')} -> {session_config.get('output_audio_format', 'N/A')}")
            
            elif event_type == 'response.audio.delta':
                # Accumulate audio chunks
                delta = data.get('delta', '')
                if delta:
                    # Convert from base64 to bytes (not hex)
                    import base64
                    audio_chunk = base64.b64decode(delta)
                    self.audio_buffer.extend(audio_chunk)
                    print(f"🎵 Received audio chunk: {len(audio_chunk)} bytes (total: {len(self.audio_buffer)} bytes)")
            
            elif event_type == 'response.audio.done':
                # Audio response complete
                print("🎵 Audio response complete")
                # Don't set complete here - wait for response.done
            
            elif event_type == 'response.done':
                # Full response complete
                print("✅ Response generation complete")
                # Check if we got any audio
                if len(self.audio_buffer) == 0:
                    print("⚠️  No audio data received in response")
                self.response_complete.set()
            
            elif event_type == 'response.text.delta':
                # Text response (shouldn't happen with audio modality)
                text_delta = data.get('delta', '')
                print(f"📝 Received text instead of audio: {text_delta}")
            
            elif event_type == 'response.text.done':
                # Complete text response
                text = data.get('text', '')
                print(f"📝 Complete text response: {text}")
            
            elif event_type == 'session.updated':
                print("✅ Session configuration updated")
            
            elif event_type == 'error':
                error_data = data.get('error', {})
                error_type = error_data.get('type', 'unknown_error')
                error_code = error_data.get('code', 'N/A')
                error_message = error_data.get('message', 'Unknown error')
                print(f"❌ API Error [{error_type}:{error_code}]: {error_message}")
                self.response_complete.set()
            
            elif event_type == 'input_audio_buffer.speech_started':
                print("🎤 Speech detected in input")
            
            elif event_type == 'input_audio_buffer.speech_stopped':
                print("🔇 Speech ended in input")
            
            elif event_type == 'input_audio_buffer.committed':
                item_id = data.get('item_id', 'unknown')
                print(f"✅ Audio buffer committed (item_id: {item_id})")
            
            elif event_type == 'response.created':
                print("📝 Response creation started")
            
            elif event_type == 'input_audio_buffer.cleared':
                print("🗑️  Audio buffer cleared")
            
            elif event_type == 'conversation.item.input_audio_transcription.completed':
                # Input audio transcription
                transcript = data.get('transcript', '')
                self.input_transcript = transcript
                print(f"🎤 Input transcript: {transcript}")
            
            elif event_type == 'response.audio_transcript.delta':
                # Output audio transcript delta
                delta = data.get('delta', '')
                self.output_transcript += delta
            
            elif event_type == 'response.audio_transcript.done':
                # Output audio transcript complete
                transcript = data.get('transcript', '')
                if transcript:
                    self.output_transcript = transcript
                print(f"💬 Output transcript: {self.output_transcript}")
            
            # Debug: print other event types
            elif event_type not in ['response.content_part.added', 
                                   'conversation.item.created', 'response.content_part.done',
                                   'rate_limits.updated', 'conversation.item.input_audio_transcription.delta']:
                print(f"📨 Other event: {event_type}")
                # For debugging, show more details for certain events
                if event_type in ['response.output_item.added', 'response.output_item.done']:
                    print(f"   Details: {json.dumps(data, indent=2)[:200]}...")
                    
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            print(f"Raw message: {message[:200]}...")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"❌ WebSocket error: {error}")
        self.response_queue.put({'type': 'error', 'error': str(error)})
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print("🔌 Disconnected from OpenAI Realtime API")


class CartesiaOpenAIPipeline:
    """Pipeline for Cartesia STT -> OpenAI GPT-4 -> Cartesia TTS."""
    
    def __init__(self, cartesia_api_key: str, openai_api_key: str):
        self.cartesia_api_key = cartesia_api_key
        self.openai_api_key = openai_api_key
        self.cartesia_base_url = "https://api.cartesia.ai"
        self.last_stt_duration = 0  # For cost calculation
        
    def process_audio(self, audio_file: str) -> Tuple[Optional[bytes], float, dict]:
        """Process audio through the Cartesia-OpenAI pipeline."""
        start_time = time.time()
        cost_info = {
            'stt_cost': 0,
            'llm_cost': 0,
            'tts_cost': 0,
            'total_cost': 0,
            'stt_time': 0,
            'llm_time': 0,
            'tts_time': 0,
            'transcript': '',
            'llm_response': '',
            'error': None
        }
        
        try:
            # Step 1: Speech-to-Text with Cartesia
            stt_start = time.time()
            transcript = self._speech_to_text(audio_file)
            cost_info['stt_time'] = time.time() - stt_start
            cost_info['transcript'] = transcript
            
            if not transcript:
                raise Exception("Failed to transcribe audio")
            
            print(f"📝 Transcript: {transcript}")
            
            # Step 2: Process with OpenAI GPT-4
            llm_start = time.time()
            llm_response, llm_tokens = self._process_with_gpt4(transcript)
            cost_info['llm_time'] = time.time() - llm_start
            cost_info['llm_response'] = llm_response
            
            if not llm_response:
                raise Exception("Failed to get GPT-4 response")
            
            print(f"🤖 GPT-4o response: {llm_response}")
            
            # Calculate LLM cost (GPT-4o pricing)
            cost_info['llm_input_tokens'] = llm_tokens['input']
            cost_info['llm_output_tokens'] = llm_tokens['output']
            cost_info['llm_cost'] = (
                (llm_tokens['input'] / 1_000_000) * 2.50 +  # $2.50 per 1M input tokens
                (llm_tokens['output'] / 1_000_000) * 10.00  # $10.00 per 1M output tokens
            )
            
            # Step 3: Text-to-Speech with Cartesia
            tts_start = time.time()
            audio_response = self._text_to_speech(llm_response)
            cost_info['tts_time'] = time.time() - tts_start
            
            if not audio_response:
                raise Exception("Failed to synthesize speech")
            
            # Calculate Cartesia costs
            # WARNING: These are estimates based on available pricing info
            # STT (Ink-Whisper): 1 credit per second of audio
            # Using Pro tier pricing: $5 for 100K credits = $0.00005 per credit
            if hasattr(self, 'last_stt_duration'):
                stt_credits = self.last_stt_duration  # 1 credit per second
                cost_info['stt_cost'] = stt_credits * 0.00005  # Pro tier pricing
            else:
                # Fallback: estimate 150 words per minute speech rate
                estimated_seconds = (len(transcript.split()) / 150) * 60
                cost_info['stt_cost'] = estimated_seconds * 0.00005
            
            # TTS (Sonic): 1 credit per character of INPUT text
            # Note: It takes 750-800 credits to generate 1 minute of audio
            tts_credits = len(llm_response)  # 1 credit per character
            cost_info['tts_cost'] = tts_credits * 0.00005  # Pro tier pricing
            
            # For reference: audio generation rate
            # ~150 words/minute speaking rate, ~5 chars/word = ~750 chars/minute
            # This aligns with Cartesia's 750-800 credits per minute of audio
            
            # Calculate total cost
            cost_info['total_cost'] = (
                cost_info['stt_cost'] + 
                cost_info['llm_cost'] + 
                cost_info['tts_cost']
            )
            
            # Add transcripts
            cost_info['input_transcript'] = transcript
            cost_info['output_transcript'] = llm_response
            
            # Stop timer here - before any playback
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"⏱️  Total pipeline time: {elapsed_time:.2f}s (playback not included)")
            
            return audio_response, elapsed_time, cost_info
            
        except Exception as e:
            cost_info['error'] = str(e)
            print(f"❌ Pipeline error: {e}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            return None, elapsed_time, cost_info
    
    def _speech_to_text(self, audio_file: str) -> Optional[str]:
        """Convert speech to text using Cartesia."""
        try:
            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": self.cartesia_api_key
            }
            
            with open(audio_file, 'rb') as f:
                files = {'file': ('audio.wav', f, 'audio/wav')}
                data = {
                    'model': 'ink-whisper',
                    'language': 'en',
                    'timestamp_granularities[]': 'word'  # Optional, for word timestamps
                }
                
                # Using the official Cartesia STT endpoint from docs
                response = requests.post(
                    f"{self.cartesia_base_url}/stt",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                # Store duration for cost calculation
                if hasattr(self, 'last_stt_duration'):
                    self.last_stt_duration = result.get('duration', 0)
                return result.get('text', '')  # Note: API returns 'text' not 'transcript'
            else:
                print(f"❌ Cartesia STT error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ STT error: {e}")
            return None
    
    def _process_with_gpt4(self, text: str) -> Tuple[Optional[str], dict]:
        """Process text with OpenAI GPT-4o."""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Respond concisely to the user's message."},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content']
                tokens = {
                    'input': result['usage']['prompt_tokens'],
                    'output': result['usage']['completion_tokens']
                }
                return message, tokens
            else:
                print(f"❌ GPT-4o error: {response.status_code} - {response.text}")
                return None, {'input': 0, 'output': 0}
                
        except Exception as e:
            print(f"❌ GPT-4o error: {e}")
            return None, {'input': 0, 'output': 0}
    
    def _text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Cartesia."""
        try:
            print("🔧 Using Cartesia direct API for TTS...")
            
            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": self.cartesia_api_key,
                "Content-Type": "application/json"
            }
            
            data = {    
                "model_id": "sonic-2",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": "694f9389-aac1-45b6-b726-9d9369183238"
                },
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 24000
                },
                "language": "en"
            }
            
            response = requests.post(
                f"{self.cartesia_base_url}/tts/bytes",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                audio_data = response.content
                print(f"📊 Cartesia TTS response: {len(audio_data)} bytes")
                expected_duration = len(audio_data) / (24000 * 2)
                print(f"   Expected duration: {expected_duration:.2f} seconds")
                return audio_data
            else:
                print(f"❌ Cartesia TTS error: {response.status_code} - {response.text}")
                return None
                    
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None


def save_audio_response(audio_data: bytes, filename: str, sample_rate: int = 24000) -> None:
    """Save raw audio data to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


def print_results(method: str, elapsed_time: float, cost_info: dict) -> None:
    """Print formatted results for a test."""
    print(f"\n{'='*60}")
    print(f"📊 {method} Results")
    print(f"{'='*60}")
    print(f"⏱️  Response Time: {elapsed_time:.2f} seconds")
    
    if 'error' in cost_info and cost_info['error']:
        print(f"❌ Error: {cost_info['error']}")
    
    if method == "OpenAI Realtime API":
        print(f"📥 Input Tokens: {cost_info.get('input_tokens', 0):,}")
        print(f"📤 Output Tokens: {cost_info.get('output_tokens', 0):,}")
        print(f"💰 Input Cost: ${cost_info.get('input_cost', 0):.4f}")
        print(f"💰 Output Cost: ${cost_info.get('output_cost', 0):.4f}")
        print(f"💰 Total Cost: ${cost_info.get('total_cost', 0):.4f}")
        
        # Show transcripts if available
        if 'input_transcript' in cost_info and cost_info['input_transcript']:
            print(f"\n🎤 What you said: \"{cost_info['input_transcript']}\"")
        if 'output_transcript' in cost_info and cost_info['output_transcript']:
            print(f"💬 AI response: \"{cost_info['output_transcript']}\"")
    else:
        print(f"\n⏱️  Breakdown:")
        print(f"   - Speech-to-Text: {cost_info.get('stt_time', 0):.2f}s")
        print(f"   - GPT-4 Processing: {cost_info.get('llm_time', 0):.2f}s")
        print(f"   - Text-to-Speech: {cost_info.get('tts_time', 0):.2f}s")
        
        print(f"\n💰 Cost Breakdown:")
        print(f"   - STT Cost: ${cost_info.get('stt_cost', 0):.4f}")
        print(f"   - LLM Cost: ${cost_info.get('llm_cost', 0):.4f}")
        print(f"     • Input tokens: {cost_info.get('llm_input_tokens', 0):,}")
        print(f"     • Output tokens: {cost_info.get('llm_output_tokens', 0):,}")
        print(f"   - TTS Cost: ${cost_info.get('tts_cost', 0):.4f}")
        print(f"   - Total Cost: ${cost_info.get('total_cost', 0):.4f}")
        
        # Show transcripts if available
        if 'input_transcript' in cost_info and cost_info['input_transcript']:
            print(f"\n🎤 What you said: \"{cost_info['input_transcript']}\"")
        if 'output_transcript' in cost_info and cost_info['output_transcript']:
            print(f"💬 AI response: \"{cost_info['output_transcript']}\"")


def main():
    """Main function to run the audio model comparison."""
    print("\n🎯 Audio Model Comparison Test")
    print("="*60)
    
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    
    if not openai_api_key:
        print("❌ Please set the OPENAI_API_KEY environment variable")
        return
    
    if not cartesia_api_key:
        print("❌ Please set the CARTESIA_API_KEY environment variable")
        return
    
    # Initialize components
    recorder = AudioRecorder(duration=5, sample_rate=24000)
    player = AudioPlayer()
    
    # Record audio input
    input_filename = "input_audio.wav"
    recorder.record(input_filename)
    
    print("\n🚀 Starting tests...")
    
    # Test 1: OpenAI Realtime API
    print("\n📡 Testing OpenAI Realtime API...")
    realtime_client = OpenAIRealtimeClient(openai_api_key)
    realtime_audio, realtime_time, realtime_cost = realtime_client.process_audio(input_filename)
    
    if realtime_audio:
        realtime_response_file = "openai_realtime_response.wav"
        save_audio_response(realtime_audio, realtime_response_file)
        print("\n🔊 Playing OpenAI Realtime response immediately...")
        player.play_file(realtime_response_file)
        print("✅ OpenAI Realtime response completed")
    else:
        print("❌ No audio received from OpenAI Realtime API")
    
    # Brief pause before starting the next test
    print("\n⏸️  Pausing for 2 seconds before next test...")
    time.sleep(2)
    
    # Now proceed with Cartesia pipeline
    print("\n" + "="*60)
    print("🔄 Now testing Cartesia + GPT-4o Pipeline...")
    print("="*60)
    
    pipeline = CartesiaOpenAIPipeline(cartesia_api_key, openai_api_key)
    pipeline_audio, pipeline_time, pipeline_cost = pipeline.process_audio(input_filename)
    
    if pipeline_audio:
        pipeline_response_file = "cartesia_pipeline_response.wav"
        save_audio_response(pipeline_audio, pipeline_response_file)
        print("\n🔊 Playing Cartesia + GPT-4 response...")
        player.play_file(pipeline_response_file)
        print("✅ Cartesia pipeline response completed")
    else:
        print("❌ No audio received from Cartesia pipeline")
    
    # Print results
    print_results("OpenAI Realtime API", realtime_time, realtime_cost)
    print_results("Cartesia + GPT-4 Pipeline", pipeline_time, pipeline_cost)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📈 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if realtime_time > 0 and pipeline_time > 0:
        time_diff = abs(realtime_time - pipeline_time)
        faster_method = "OpenAI Realtime" if realtime_time < pipeline_time else "Cartesia Pipeline"
        time_ratio = max(realtime_time, pipeline_time) / min(realtime_time, pipeline_time)
        
        print(f"\n⏱️  Speed: {faster_method} was {time_ratio:.1f}x faster")
        print(f"   - OpenAI Realtime: {realtime_time:.2f}s")
        print(f"   - Cartesia Pipeline: {pipeline_time:.2f}s")
        print(f"   - Difference: {time_diff:.2f}s")
        
        realtime_total_cost = realtime_cost.get('total_cost', 0)
        pipeline_total_cost = pipeline_cost.get('total_cost', 0)
        
        if realtime_total_cost > 0 and pipeline_total_cost > 0:
            cheaper_method = "OpenAI Realtime" if realtime_total_cost < pipeline_total_cost else "Cartesia Pipeline"
            cost_ratio = max(realtime_total_cost, pipeline_total_cost) / min(realtime_total_cost, pipeline_total_cost)
            
            print(f"\n💰 Cost: {cheaper_method} was {cost_ratio:.1f}x cheaper")
            print(f"   - OpenAI Realtime: ${realtime_total_cost:.4f}")
            print(f"   - Cartesia Pipeline: ${pipeline_total_cost:.4f}")
            print(f"   - Difference: ${abs(realtime_total_cost - pipeline_total_cost):.4f}")
    
    print(f"\n✅ Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
