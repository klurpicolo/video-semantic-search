import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import tempfile
import moviepy as mp
import os

# --- Configuration ---
# Use a standard CLIP model as the encoder foundation
MODEL_NAME = "openai/clip-vit-base-patch32"
FRAME_COUNT = 32  # Number of frames to sample from the video
EMBEDDING_DIM = 512  # Dimension of CLIP-ViT-B/32 embedding
CLIP_MODEL_URL = "https://huggingface.co/openai/clip-vit-base-patch32"


# --- Initialization ---

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor."""
    with st.spinner("Loading CLIP model... This happens once."):
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model.eval()
    return model, processor


# Load model globally
clip_model, clip_processor = load_clip_model()

# Global state for the vector database
if 'video_db' not in st.session_state:
    st.session_state.video_db = {}
    st.session_state.video_ids = []


# --- Core Video Processing and Alignment ---

def get_video_embeddings_and_temporal_info(video_path, num_frames=FRAME_COUNT):
    """
    Extracts frame features and a global feature for a video.
    Returns: Global (aggregated) embedding, a 3D tensor of frame embeddings, and frame timestamps.
    """
    try:
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration

        # Calculate time points for frame sampling
        frame_timestamps = np.linspace(0, duration, num_frames, endpoint=False)

        frame_embeddings = []

        for t in frame_timestamps:
            frame_pil = Image.fromarray(clip.get_frame(t))
            inputs = clip_processor(images=frame_pil, return_tensors="pt")

            with torch.no_grad():
                # Get the embedding for the frame
                frame_features = clip_model.get_image_features(inputs["pixel_values"])

            # Normalize and append
            frame_embeddings.append(frame_features / frame_features.norm(dim=-1, keepdim=True))

        # (N_frames, 1, D) -> (N_frames, D)
        frame_embeddings_tensor = torch.stack(frame_embeddings).squeeze(1)

        # Global Video Feature (for overall relevance)
        # In the original X-CLIP, a Transformer refines this, but here we use mean pooling
        global_embedding = frame_embeddings_tensor.mean(dim=0)

        return global_embedding.cpu().numpy(), frame_embeddings_tensor.cpu().numpy(), frame_timestamps

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None, None


def get_text_embedding(text):
    """Computes the CLIP embedding for a text query."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    # Normalize
    return (text_features / text_features.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()


def find_best_moment(text_embedding, frame_embeddings, timestamps, moment_duration=5.0):
    """
    Simulates X-CLIP's local alignment by finding the frame most similar to the query.

    Returns: (start_time, end_time, max_similarity)
    """
    # frame_embeddings shape: (N_frames, D)
    # text_embedding shape: (D,)

    # 1. Calculate similarity score for every frame
    # (N_frames, D) dot (D,) -> (N_frames,)
    frame_similarities = np.dot(frame_embeddings, text_embedding)

    # 2. Find the frame with the highest similarity
    best_frame_index = np.argmax(frame_similarities)
    max_similarity = frame_similarities[best_frame_index]

    # 3. Determine the 5-second moment around that frame
    best_time = timestamps[best_frame_index]

    # Calculate half the moment duration
    half_moment = moment_duration / 2

    # Ensure the moment stays within the video boundaries
    start_time = max(0.0, best_time - half_moment)
    end_time = min(timestamps[-1] + (timestamps[1] - timestamps[0]), best_time + half_moment)

    return start_time, end_time, max_similarity


# --- Streamlit UI ---

st.title("🎬 X-CLIP Style Video Retrieval with Temporal Grounding")
st.markdown(f"Using **{MODEL_NAME}** for feature extraction and **Frame-to-Text alignment** for temporal matching.")
st.markdown("---")

## 1. Build/Update Video Database
st.header("1. Video Database Indexing")
uploaded_file = st.file_uploader("Upload an MP4 Video:", type="mp4")

if uploaded_file is not None:
    video_name = uploaded_file.name

    if video_name not in st.session_state.video_db:
        progress_bar = st.progress(0, text=f"Processing {video_name}...")

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # 1. Get Embeddings
        global_embed, frame_embeds, timestamps = get_video_embeddings_and_temporal_info(temp_path,
                                                                                        num_frames=FRAME_COUNT)

        if global_embed is not None:
            # 2. Update database
            st.session_state.video_db[video_name] = {
                'path': temp_path,
                'global_embedding': global_embed,
                'frame_embeddings': frame_embeds,
                'timestamps': timestamps
            }
            # 3. Update ID list
            st.session_state.video_ids = list(st.session_state.video_db.keys())

            progress_bar.progress(100, text=f"Processing {video_name}... Done!")
            st.success(f"Video '{video_name}' added and indexed ({FRAME_COUNT} frames).")
        else:
            os.unlink(temp_path)  # Clean up temp file on error

# Display current database status
if st.session_state.video_db:
    st.info(f"Database Status: **{len(st.session_state.video_db)}** videos indexed.")
else:
    st.warning("Database is empty. Please upload a video to begin.")

st.markdown("---")

## 2. Search and Temporal Grounding
st.header("2. Semantic Search and Moment Retrieval")

query_text = st.text_input(
    "Enter your text query:",
    placeholder="e.g., A slow-motion shot of a cat jumping"
)

search_button = st.button("Search Videos & Find Moments", type="primary", disabled=not st.session_state.video_db)

if search_button and query_text:

    # 1. Get text query embedding (Global Text Feature)
    with st.spinner(f"Encoding query: '{query_text}'..."):
        query_embedding = get_text_embedding(query_text)

    st.subheader(f"Results for: **'{query_text}'**")

    # List to hold video scores and details
    video_scores = []

    # 2. Score All Videos (Global Retrieval)
    for video_name, data in st.session_state.video_db.items():
        # Global Similarity (Video-level match)
        global_similarity = np.dot(data['global_embedding'], query_embedding)

        # Temporal Grounding (Local Frame Match)
        start_t, end_t, max_local_sim = find_best_moment(
            query_embedding,
            data['frame_embeddings'],
            data['timestamps']
        )

        video_scores.append({
            'name': video_name,
            'global_similarity': global_similarity,
            'start_time': start_t,
            'end_time': end_t,
            'max_local_similarity': max_local_sim,
            'path': data['path']
        })

    # Sort by the global similarity score (overall video relevance)
    ranked_videos = sorted(video_scores, key=lambda x: x['global_similarity'], reverse=True)

    # 3. Display Results
    for rank, result in enumerate(ranked_videos[:5]):  # Show top 5 videos

        st.markdown(f"#### Rank {rank + 1}: {result['name']}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**Overall Relevance (Global):** {result['global_similarity']:.4f}")
            st.markdown(f"**Best Moment Relevance (Local):** {result['max_local_similarity']:.4f}")

            # Show the best 5-second window
            st.markdown(f"**📺 Best Moment Found:** `{result['start_time']:.1f}s` to `{result['end_time']:.1f}s`")

        with col2:
            st.video(
                result['path'],
                start_time=int(result['start_time']),
                # Note: 'moviepy' processes are expensive, limit the displayed segment length
            )
            st.caption("Video starts at the most relevant moment.")

        st.markdown("---")