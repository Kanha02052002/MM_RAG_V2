
import streamlit as st
import requests
import base64
import json
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import tempfile
from PIL import Image
import io

# Configuration
@dataclass
class Config:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    VISION_MODEL = "google/gemini-pro-vision"
    TEXT_MODEL = "anthropic/claude-3-sonnet"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_COLLECTION = "openrouter_responses"

# Initialize session state
def init_session_state():
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = None
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

# Embedding and Vector Store Classes
class HuggingFaceEmbedding:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> List[float]:
        if self.model is None:
            return [0.0] * 384  # Default dimension
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            st.error(f"Error creating embedding: {e}")
            return [0.0] * 384

class ChromaVectorStore:
    def __init__(self, collection_name: str = Config.CHROMA_COLLECTION):
        try:
            # Use in-memory database for Streamlit
            self.client = chromadb.Client(Settings(
                allow_reset=True,
                anonymized_telemetry=False
            ))
            self.collection_name = collection_name
            self.collection = None
            self._initialize_collection()
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
            self.client = None
    
    def _initialize_collection(self):
        try:
            if self.client:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "OpenRouter API responses"}
                )
        except Exception as e:
            st.error(f"Error creating collection: {e}")
    
    def add_document(self, text: str, metadata: Dict[str, Any], embedding: List[float]):
        if not self.collection:
            return False
        try:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            st.error(f"Error adding document: {e}")
            return False
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            st.error(f"Error searching: {e}")
            return []

# OpenRouter API Client
class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = Config.OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Streamlit OpenRouter App"
        }
    
    def encode_image(self, image_file) -> str:
        """Encode image to base64 string"""
        try:
            if isinstance(image_file, str):
                with open(image_file, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Error encoding image: {e}")
            return ""
    
    def vision_request(self, prompt: str, image_file, model: str = Config.VISION_MODEL) -> Dict:
        """Make vision API request with image"""
        try:
            # Encode image
            base64_image = self.encode_image(image_file)
            if not base64_image:
                return {"error": "Failed to encode image"}
            
            # Prepare request
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            return response.json()
        except Exception as e:
            return {"error": f"Vision API request failed: {str(e)}"}
    
    def text_request(self, prompt: str, model: str = Config.TEXT_MODEL, structured: bool = False) -> Dict:
        """Make text API request"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if structured:
                # Add structured response instructions
                structured_prompt = f"""
                Please provide a structured response to the following query in JSON format:
                
                {{
                    "summary": "Brief summary of the response",
                    "main_points": ["Key point 1", "Key point 2", "..."],
                    "detailed_response": "Detailed explanation",
                    "confidence_level": "High/Medium/Low",
                    "sources_needed": ["Any sources that would be helpful"],
                    "follow_up_questions": ["Relevant follow-up questions"]
                }}
                
                Query: {prompt}
                """
                messages[0]["content"] = structured_prompt
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            return response.json()
        except Exception as e:
            return {"error": f"Text API request failed: {str(e)}"}

# Structured Response Parser
def parse_structured_response(response_text: str) -> Dict:
    """Parse structured JSON response"""
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            # Fallback to basic structure
            return {
                "summary": "Response parsing failed",
                "main_points": [],
                "detailed_response": response_text,
                "confidence_level": "Unknown",
                "sources_needed": [],
                "follow_up_questions": []
            }
    except Exception as e:
        return {
            "summary": "JSON parsing failed",
            "main_points": [],
            "detailed_response": response_text,
            "confidence_level": "Unknown",
            "sources_needed": [],
            "follow_up_questions": [],
            "parsing_error": str(e)
        }

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="OpenRouter Vision & Vector Store App",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("OpenRouter Vision & Vector Store App")
    st.markdown("*Powered by OpenRouter API, ChromaDB, and HuggingFace Embeddings*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.api_key,
            help="Get your API key from OpenRouter"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        # Initialize components
        if st.button("Initialize Components"):
            if api_key:
                with st.spinner("Initializing..."):
                    # Initialize embedding model
                    st.session_state.embedding_model = HuggingFaceEmbedding()
                    
                    # Initialize vector store
                    st.session_state.chroma_client = ChromaVectorStore()
                    
                    if st.session_state.embedding_model.model is not None:
                        st.success("✅ Components initialized successfully!")
                    else:
                        st.error("❌ Failed to initialize embedding model")
            else:
                st.error("Please enter your OpenRouter API key")
        
        # Model selection
        st.subheader("Model Selection")
        vision_model = st.selectbox(
            "Vision Model",
            ["google/gemini-pro-vision", "anthropic/claude-3-sonnet", "openai/gpt-4-vision-preview"],
            help="Select vision model for image analysis"
        )
        
        text_model = st.selectbox(
            "Text Model",
            ["anthropic/claude-3-sonnet", "openai/gpt-4", "google/gemini-pro"],
            help="Select text model for general queries"
        )
        
        # Vector store stats
        st.subheader("📊 Vector Store Stats")
        if st.session_state.chroma_client and st.session_state.chroma_client.collection:
            try:
                count = st.session_state.chroma_client.collection.count()
                st.metric("Stored Documents", count)
            except:
                st.metric("Stored Documents", "Unknown")
        else:
            st.metric("Stored Documents", "Not initialized")
        
        # Clear vector store
        if st.button("Clear Vector Store"):
            if st.session_state.chroma_client:
                try:
                    st.session_state.chroma_client.client.reset()
                    st.session_state.chroma_client._initialize_collection()
                    st.success("Vector store cleared!")
                except Exception as e:
                    st.error(f"Error clearing vector store: {e}")
    
    # Main content area
    if not st.session_state.api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar to get started.")
        return
    
    # Initialize client
    client = OpenRouterClient(st.session_state.api_key)
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Text Chat", "👁️ Vision Analysis", "🔍 Vector Search", "📈 Analytics"])
    
    with tab1:
        st.header("Text Chat with Structured Responses")
        
        # Text input
        text_prompt = st.text_area(
            "Enter your question or prompt:",
            placeholder="Ask anything...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            structured_response = st.checkbox("Enable Structured Response", value=True)
        with col2:
            save_to_vector = st.checkbox("Save to Vector Store", value=True)
        
        if st.button("Send Message", type="primary"):
            if text_prompt:
                with st.spinner("Processing..."):
                    # Make API request
                    response = client.text_request(
                        text_prompt, 
                        model=text_model, 
                        structured=structured_response
                    )
                    
                    if "error" not in response:
                        # Extract response content
                        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Display response
                        if structured_response:
                            parsed_response = parse_structured_response(content)
                            
                            # Display structured response
                            st.subheader("📋 Structured Response")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Summary:**")
                                st.write(parsed_response.get("summary", "N/A"))
                                
                                st.write("**Confidence Level:**")
                                confidence = parsed_response.get("confidence_level", "Unknown")
                                if confidence == "High":
                                    st.success(f"🟢 {confidence}")
                                elif confidence == "Medium":
                                    st.warning(f"🟡 {confidence}")
                                else:
                                    st.error(f"🔴 {confidence}")
                            
                            with col2:
                                st.write("**Main Points:**")
                                for point in parsed_response.get("main_points", []):
                                    st.write(f"• {point}")
                            
                            st.write("**Detailed Response:**")
                            st.write(parsed_response.get("detailed_response", ""))
                            
                            if parsed_response.get("sources_needed"):
                                st.write("**Sources Needed:**")
                                for source in parsed_response["sources_needed"]:
                                    st.write(f"• {source}")
                            
                            if parsed_response.get("follow_up_questions"):
                                st.write("**Follow-up Questions:**")
                                for question in parsed_response["follow_up_questions"]:
                                    st.write(f"• {question}")
                        else:
                            st.write(content)
                        
                        # Save to vector store
                        if save_to_vector and st.session_state.embedding_model and st.session_state.chroma_client:
                            try:
                                embedding = st.session_state.embedding_model.embed_text(content)
                                metadata = {
                                    "type": "text_response",
                                    "model": text_model,
                                    "prompt": text_prompt,
                                    "timestamp": datetime.now().isoformat(),
                                    "structured": structured_response
                                }
                                
                                success = st.session_state.chroma_client.add_document(
                                    content, metadata, embedding
                                )
                                
                                if success:
                                    st.success("💾 Response saved to vector store!")
                            except Exception as e:
                                st.error(f"Error saving to vector store: {e}")
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "type": "text",
                            "prompt": text_prompt,
                            "response": content,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        st.error(f"API Error: {response.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a prompt.")
    
    with tab2:
        st.header("Vision Analysis")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image for AI analysis"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Vision prompt
            vision_prompt = st.text_area(
                "What would you like me to analyze about this image?",
                placeholder="Describe what you see in this image...",
                height=100
            )
            
            save_vision_to_vector = st.checkbox("Save analysis to Vector Store", value=True)
            
            if st.button("Analyze Image", type="primary"):
                if vision_prompt:
                    with st.spinner("Analyzing image..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Make vision API request
                        response = client.vision_request(
                            vision_prompt,
                            uploaded_file,
                            model=vision_model
                        )
                        
                        if "error" not in response:
                            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            st.subheader("🔍 Vision Analysis Result")
                            st.write(content)
                            
                            # Save to vector store
                            if save_vision_to_vector and st.session_state.embedding_model and st.session_state.chroma_client:
                                try:
                                    embedding = st.session_state.embedding_model.embed_text(content)
                                    metadata = {
                                        "type": "vision_response",
                                        "model": vision_model,
                                        "prompt": vision_prompt,
                                        "timestamp": datetime.now().isoformat(),
                                        "image_name": uploaded_file.name
                                    }
                                    
                                    success = st.session_state.chroma_client.add_document(
                                        content, metadata, embedding
                                    )
                                    
                                    if success:
                                        st.success("💾 Analysis saved to vector store!")
                                except Exception as e:
                                    st.error(f"Error saving to vector store: {e}")
                            
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                "type": "vision",
                                "prompt": vision_prompt,
                                "response": content,
                                "timestamp": datetime.now().isoformat(),
                                "image_name": uploaded_file.name
                            })
                        else:
                            st.error(f"Vision API Error: {response.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter a prompt for image analysis.")
    
    with tab3:
        st.header("Vector Search")
        
        if not st.session_state.embedding_model or not st.session_state.chroma_client:
            st.warning("Please initialize components in the sidebar first.")
            return
        
        # Search input
        search_query = st.text_input(
            "Search your stored responses:",
            placeholder="Enter search query..."
        )
        
        num_results = st.slider("Number of results", 1, 10, 5)
        
        if st.button("Search", type="primary"):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        # Create embedding for search query
                        query_embedding = st.session_state.embedding_model.embed_text(search_query)
                        
                        # Search vector store
                        results = st.session_state.chroma_client.search(query_embedding, num_results)
                        
                        if results and 'documents' in results and results['documents'][0]:
                            st.subheader(f"🔍 Found {len(results['documents'][0])} results")
                            
                            for i, (doc, metadata, distance) in enumerate(zip(
                                results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]
                            )):
                                with st.expander(f"Result {i+1} (Similarity: {1-distance:.3f})"):
                                    st.write("**Content:**")
                                    st.write(doc)
                                    
                                    st.write("**Metadata:**")
                                    for key, value in metadata.items():
                                        st.write(f"• **{key}:** {value}")
                        else:
                            st.info("No results found.")
                    except Exception as e:
                        st.error(f"Search error: {e}")
            else:
                st.warning("Please enter a search query.")
    
    with tab4:
        st.header("Analytics & History")
        
        # Conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            # Filter options
            filter_type = st.selectbox(
                "Filter by type:",
                ["All", "Text", "Vision"]
            )
            
            filtered_history = st.session_state.conversation_history
            if filter_type != "All":
                filtered_history = [
                    item for item in st.session_state.conversation_history 
                    if item["type"].lower() == filter_type.lower()
                ]
            
            # Display history
            for i, item in enumerate(reversed(filtered_history)):
                with st.expander(f"{item['type'].title()} - {item['timestamp']}"):
                    st.write("**Prompt:**")
                    st.write(item["prompt"])
                    st.write("**Response:**")
                    st.write(item["response"])
                    
                    if item["type"] == "vision" and "image_name" in item:
                        st.write(f"**Image:** {item['image_name']}")
        else:
            st.info("No conversation history yet. Start chatting or analyzing images!")
        
        # Clear history
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.success("History cleared!")

if __name__ == "__main__":
    main()
