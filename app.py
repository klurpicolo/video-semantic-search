import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import io
import tempfile
import moviepy as mp

# --- Configuration ---
# Use a smaller CLIP model for faster local inference
MODEL_NAME = "openai/clip-vit-base-patch32"


# --- Initialization ---

# @st.cache_resource is the best way to load large models in Streamlit
@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor only once."""
    st.write("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    # Set model to evaluation mode
    model.eval()
    return model, processor


# Load model globally
clip_model, clip_processor = load_clip_model()

# Global state for embeddings (simple in-memory database)
if 'video_db' not in st.session_state:
    st.session_state.video_db = {}
    st.session_state.embeddings = None


# --- Core Video Processing Function ---

def get_video_embedding(video_path, num_frames=16):
    """
    Extracts frames from a video and computes a single, aggregated CLIP embedding.
    This simulates the core idea of CLIP4Clip (frame sampling + aggregation).
    """
    try:
        # Load the video clip
        clip = mp.VideoFileClip(video_path)

        # Calculate time points for frame sampling
        duration = clip.duration
        frame_indices = np.linspace(0, duration, num_frames, endpoint=False)

        frame_embeddings = []

        # Process each sampled frame
        for t in frame_indices:
            # Extract frame at time t as a PIL image
            frame_pil = Image.fromarray(clip.get_frame(t))

            # Preprocess the frame
            inputs = clip_processor(images=frame_pil, return_tensors="pt", padding=True)

            # Compute image features (embedding)
            with torch.no_grad():
                image_features = clip_model.get_image_features(inputs["pixel_values"])

            # Normalize and store
            frame_embeddings.append(image_features / image_features.norm(dim=-1, keepdim=True))

        # Aggregate (mean pooling) to get a single video embedding (CLIP4Clip style)
        video_embedding = torch.stack(frame_embeddings).mean(dim=0)

        # Return as a numpy array for the database
        return video_embedding.squeeze().cpu().numpy()

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


# --- Core Retrieval Function ---

def get_text_embedding(text):
    """Computes the CLIP embedding for a text query."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    # Normalize
    return (text_features / text_features.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()


def perform_retrieval(query_embedding):
    """Performs a cosine similarity search against the video database."""
    db_embeddings = st.session_state.embeddings

    if db_embeddings is None:
        return []

    # Calculate cosine similarity: dot product of normalized vectors
    # (Query is 1D, DB is 2D: (N, D))
    similarities = np.dot(db_embeddings, query_embedding)

    # Get indices of the videos sorted by similarity (highest first)
    sorted_indices = np.argsort(similarities)[::-1]

    # Map indices back to video IDs and similarity scores
    results = []
    video_ids = list(st.session_state.video_db.keys())
    for i in sorted_indices:
        vid_id = video_ids[i]
        results.append({
            'id': vid_id,
            'similarity': similarities[i]
        })
    return results


# --- Streamlit UI ---

st.title("🎥 Local Video-Semantic Search (CLIP4Clip Concept)")
st.caption("Embeds video frames using CLIP and searches using cosine similarity.")

# --- Section 1: Upload and Embed Videos ---

st.header("1. Build/Update Video Database")
uploaded_file = st.file_uploader("Upload an MP4 Video:", type="mp4")

if uploaded_file is not None:
    video_name = uploaded_file.name

    if video_name not in st.session_state.video_db:
        progress_bar = st.progress(0, text=f"Processing {video_name}...")

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Get the video embedding
        video_embedding = get_video_embedding(temp_path, num_frames=16)

        if video_embedding is not None:
            # Update database
            st.session_state.video_db[video_name] = {'path': temp_path, 'embedding': video_embedding}

            # Rebuild the main embedding matrix for fast retrieval
            st.session_state.embeddings = np.array([
                data['embedding']
                for data in st.session_state.video_db.values()
            ])

            progress_bar.progress(100, text=f"Processing {video_name}... Done!")
            st.success(f"Video '{video_name}' added to the database!")
        else:
            os.unlink(temp_path)  # Clean up temp file on error

# Display current database status
if st.session_state.video_db:
    st.markdown(f"**Database Status:** {len(st.session_state.video_db)} videos indexed.")
else:
    st.info("Database is empty. Please upload a video to begin.")

st.divider()

# --- Section 2: Query the Database ---

st.header("2. Search Videos with Text")

query_text = st.text_input(
    "Enter your search query:",
    placeholder="e.g., A person riding a bike on a sunny day"
)

search_button = st.button("Search Database", type="primary", disabled=not st.session_state.video_db)

if search_button and query_text:

    # 1. Get query embedding
    query_embedding = get_text_embedding(query_text)

    # 2. Perform retrieval
    st.subheader(f"Results for: '{query_text}'")
    results = perform_retrieval(query_embedding)

    if results:
        # 3. Display results
        for rank, result in enumerate(results[:3]):  # Show top 3 results
            vid_id = result['id']
            similarity = result['similarity']
            video_data = st.session_state.video_db[vid_id]

            st.markdown(f"**Rank {rank + 1}: {vid_id}** (Similarity: {similarity:.4f})")
            st.video(video_data['path'])

    else:
        st.warning("No videos in the database to search against.")

# Clean up temp files when app closes or session ends (optional, for safety)
# This requires running cleanup on application exit, which is complex in Streamlit.
# For local demos, manual cleanup of the temp directory (e.g., /tmp) is often easier.