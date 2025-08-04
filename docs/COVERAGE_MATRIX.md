# TokenX Coverage Matrix

| Provider | API Type | SDK Call | Models | Token Tracking | Cost Tracking | Notes |
|----------|----------|----------|--------|----------------|---------------|-------|
| **OpenAI** | Chat/Completion | `client.chat.completions.create()` | gpt-4o, gpt-4o-mini, o1, o3, etc. | ✅| ✅ | Caching & streaming |
| **OpenAI** | Responses | `client.responses.create()` | gpt-4.1, gpt-4o, o1, o3, etc. | ✅| ✅ | Advanced interface |
| **OpenAI** | Embeddings | `client.embeddings.create()` | text-embedding-3-small/large | ✅| ✅ | Input tokens only |
| **OpenAI** | Audio Transcription | `client.audio.transcriptions.create()` | gpt-4o-transcribe, gpt-4o-mini-transcribe | ✅| ✅ | Dual token pricing (audio + text) |
| **OpenAI** | Audio Transcription (Legacy) | `client.audio.transcriptions.create()` | whisper-1 | ❌ | ❌ | Duration-based pricing, no usage data |
| **OpenAI** | Audio Translation | `client.audio.translations.create()` | whisper-1 | ❌ | ❌ | Duration-based pricing, no usage data |
| **OpenAI** | Text-to-Speech | `client.audio.speech.create()` | gpt-4o-mini-tts | ✅| ✅ | Dual token pricing (text + audio out) |
| **OpenAI** | Text-to-Speech (Legacy) | `client.audio.speech.create()` | tts-1, tts-1-hd | ❌ | ❌ | Character-based pricing, no usage data |
| **OpenAI** | Audio Preview | Various | gpt-4o-audio-preview, gpt-4o-mini-audio-preview | ✅| ✅ | Token-based |
| **OpenAI** | Realtime Audio | Realtime API | gpt-4o-realtime-preview, gpt-4o-mini-realtime-preview | ✅| ✅ | Caching support |
| **OpenAI** | Moderation | `client.moderations.create()` | omni-moderation-latest | ❌ | ❌ | No usage data returned |
| **OpenAI** | Images | `client.images.generate()` | gpt-image-1 | ✅| ✅ | Hybrid pricing (tokens + per-image) |
| **OpenAI** | Images (Legacy) | `client.images.generate()` | dall-e-2, dall-e-3 | ❌ | ❌ | Not supported |
| **Anthropic** | Messages | `client.messages.create()` | claude-3-*, claude-3-5-*, claude-4-* | ✅| ✅ | Differentiated cache pricing |

**Legend**: ✅ 99% accurate | ❌ Not Supported yet

### Notes

1. **Dual Token Pricing**: New audio models (gpt-4o-transcribe, gpt-4o-mini-transcribe, gpt-4o-mini-tts) now use separate pricing for audio tokens vs text tokens
2. **No Estimation**: APIs without usage data (whisper-1, tts-1, moderation) have cost tracking disabled
3. **Real Usage Only**: Only APIs with actual token usage from provider responses provide cost metrics