import streamlit as st
import json
from io import StringIO
import importlib.util

st.set_page_config(
    page_title="Claude to Gemini Format Converter",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

tiktoken_available = importlib.util.find_spec("tiktoken") is not None

if tiktoken_available:
    import tiktoken
else:
    st.error("❌ Tiktoken is required for this app to function properly")
    st.info("Install with: pip install tiktoken")
    st.stop()

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #D03434;
    }
    .success-message {
        background-color: #D1F0D1;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 4px;
    }
    .download-button {
        margin-top: 1rem;
    }
    .stAlert {
        border-radius: 5px;
    }
    .token-info {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔄 Claude to Gemini Format Converter")
st.markdown("""
This tool converts Claude-style chat JSON files to Gemini's chunkedPrompt format.
Simply upload your Claude JSON file, click convert, and download the Gemini-compatible version.
""")

try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    st.error(f"Failed to initialize tiktoken encoder: {str(e)}")
    st.stop()

def count_tokens(text):
    if not text:
        return 0
    return len(enc.encode(text))

def convert_claude_to_gemini(source_data):
    if not isinstance(source_data, dict) or "chat_messages" not in source_data:
        raise ValueError("Invalid input: Expected a JSON object with 'chat_messages' field")
    
    converted = {
        "runSettings": {
            "temperature": 1.0,
            "model": "models/gemini-2.5-pro-preview-03-25",
            "topP": 0.95,
            "topK": 64,
            "maxOutputTokens": 65536,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"}
            ],
            "responseMimeType": "text/plain",
            "enableCodeExecution": False,
            "enableEnhancedCivicAnswers": True,
            "enableSearchAsATool": False,
            "enableBrowseAsATool": False,
            "enableAutoFunctionResponse": False
        },
        "systemInstruction": {},
        "chunkedPrompt": {
            "chunks": [],
            "pendingInputs": [{
                "text": "",
                "role": "user"
            }]
        }
    }

    token_stats = {
        "total_tokens": 0,
        "user_tokens": 0,
        "model_tokens": 0,
        "thinking_tokens": 0,
        "message_count": 0,
        "has_thinking": False
    }

    for msg in source_data.get("chat_messages", []):
        role = "user" if msg["sender"] == "human" else "model"
        token_stats["message_count"] += 1
        
        for segment in msg.get("content", []):
            if segment["type"] == "thinking":
                text = segment.get("thinking", "").strip()
                if text:
                    token_stats["has_thinking"] = True
                    token_count = count_tokens(text)
                    token_stats["total_tokens"] += token_count
                    token_stats["thinking_tokens"] += token_count
                    
                    converted["chunkedPrompt"]["chunks"].append({
                        "text": text,
                        "role": role,
                        "isThought": True,
                        "tokenCount": token_count
                    })

            elif segment["type"] == "text":
                text = segment.get("text", "").strip()
                if text:
                    token_count = count_tokens(text)
                    token_stats["total_tokens"] += token_count
                    
                    if role == "user":
                        token_stats["user_tokens"] += token_count
                    else:
                        token_stats["model_tokens"] += token_count
                    
                    chunk = {
                        "text": text,
                        "role": role,
                        "tokenCount": token_count
                    }
                    if role == "model":
                        chunk["finishReason"] = "STOP"
                    converted["chunkedPrompt"]["chunks"].append(chunk)

    return converted, token_stats

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Claude JSON file", type=["json"], 
                                    help="Select a Claude-format JSON file to convert")

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name, 
            "File Type": uploaded_file.type, 
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File details:")
        st.json(file_details)
        
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            source_data = json.load(stringio)
            
            metadata = {}
            for key in ["uuid", "name", "model", "created_at"]:
                if key in source_data:
                    metadata[key] = source_data[key]
                    
            if metadata:
                st.subheader("Conversation Metadata")
                st.json(metadata)
            
            if "chat_messages" not in source_data:
                st.warning("The uploaded file doesn't appear to be a standard Claude format. It's missing the 'chat_messages' field.")
            
            # Check if the file contains thinking segments
            has_thinking = False
            for msg in source_data.get("chat_messages", []):
                for segment in msg.get("content", []):
                    if segment.get("type") == "thinking":
                        has_thinking = True
                        break
                if has_thinking:
                    break
            
            format_info = "Has thinking segments: " + ("Yes" if has_thinking else "No")
            
            st.subheader("Source Data Preview")
            message_count = len(source_data.get("chat_messages", []))
            st.write(f"Number of messages: {message_count} | {format_info}")
            
            if message_count > 0:
                first_msg = source_data["chat_messages"][0]
                st.write("First message sample:")
                st.code(json.dumps(first_msg, indent=2)[:500] + "..." if len(json.dumps(first_msg, indent=2)) > 500 else json.dumps(first_msg, indent=2))
            
            if st.button("Convert to Gemini Format", key="convert_button"):
                with st.spinner("Converting..."):
                    converted_data, token_stats = convert_claude_to_gemini(source_data)
                    
                    st.subheader("Converted Data Preview")
                    chunk_count = len(converted_data["chunkedPrompt"]["chunks"])
                    st.write(f"Number of chunks: {chunk_count}")
                    
                    st.markdown('<div class="token-info">', unsafe_allow_html=True)
                    st.write("**Token Statistics:**")
                    st.write(f"- Total tokens: {token_stats['total_tokens']:,}")
                    st.write(f"- User messages: {token_stats['user_tokens']:,} tokens")
                    st.write(f"- Model responses: {token_stats['model_tokens']:,} tokens")
                    
                    if token_stats["has_thinking"]:
                        st.write(f"- Thinking segments: {token_stats['thinking_tokens']:,} tokens")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if chunk_count > 0:
                        st.write("First chunk sample:")
                        st.code(json.dumps(converted_data["chunkedPrompt"]["chunks"][0], indent=2))
                    
                    st.markdown('<div class="success-message">✅ Conversion complete! Click the download button below.</div>', unsafe_allow_html=True)
                    
                    converted_json = json.dumps(converted_data, indent=2)
                    
                    output_filename = uploaded_file.name.replace(".json", "_gemini.json")
                    if not output_filename.endswith("_gemini.json"):
                        output_filename += "_gemini.json"
                    
                    st.markdown('<div class="download-button">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Converted File",
                        data=converted_json,
                        file_name=output_filename,
                        mime="application/json",
                        key="download_button"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        except json.JSONDecodeError:
            st.error("Error: The uploaded file is not a valid JSON file. Please check the file and try again.")
        except KeyError as e:
            st.error(f"Error: Missing required field - {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("If this problem persists, please check that your JSON follows the Claude format.")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    ### How to use:
    1. Upload a Claude-format JSON file using the uploader
    2. Click the "Convert to Gemini Format" button
    3. Review the conversion summary
    4. Download the converted file
    
    ### JSON Format Requirements:
    Your input file should be a Claude JSON with the structure:
    ```json
    {
      "uuid": "...",
      "name": "Conversation Name",
      "model": "claude-3-7-sonnet-20250219",
      "chat_messages": [
        {
          "sender": "human" or "assistant",
          "content": [
            {"type": "text", "text": "..."},
            {"type": "thinking", "thinking": "..."}
          ]
        }
      ]
    }
    ```
    
    ### Gemini Output Format:
    The converted file will follow Gemini's chunkedPrompt format:
    ```json
    {
      "runSettings": {...},
      "systemInstruction": {},
      "chunkedPrompt": {
        "chunks": [
          {"text": "...", "role": "user", "tokenCount": 123},
          {"text": "...", "role": "model", "tokenCount": 456, "finishReason": "STOP"}
        ],
        "pendingInputs": [{"text": "", "role": "user"}]
      }
    }
    ```
    """)

    st.markdown("---")
    st.markdown("© 2025 - Claude to Gemini Converter")
    st.markdown("Created with ❤️ using Streamlit")

st.markdown("---")
st.markdown("Questions or issues? File them on the GitHub repository.")
