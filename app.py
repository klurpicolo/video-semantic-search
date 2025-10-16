import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import tempfile
import moviepy as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing

# --- Configuration ---
# Use a standard CLIP model as the encoder foundation
MODEL_NAME = "openai/clip-vit-base-patch32"
FRAME_COUNT = 32  # Number of frames to sample from the video
EMBEDDING_DIM = 512  # Dimension of CLIP-ViT-B/32 embedding
CLIP_MODEL_URL = "https://huggingface.co/openai/clip-vit-base-patch32"
DATA_FOLDER = "./data"  # The folder containing the videos to ingest


# --- Initialization ---

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor."""
    with st.spinner("Loading CLIP model..."):
        # Note: torch.bfloat16 could be used for faster inference on supported GPUs
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
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = {}  # To keep track of temp files created during ingestion


# --- Core Video Processing and Alignment ---

def get_video_embeddings_and_temporal_info(video_path, num_frames=FRAME_COUNT):
    """
    Extracts frame features and a global feature for a video.
    Returns: Global (aggregated) embedding, a 3D tensor of frame embeddings, and frame timestamps.
    """
    try:
        # Check if the file is a valid video before proceeding
        if not os.path.exists(video_path):
            # This should not happen for a file in /data or a saved temp file
            raise FileNotFoundError(f"Video file not found at {video_path}")

        clip = mp.VideoFileClip(video_path)
        duration = clip.duration

        # Handle very short videos gracefully (might need better logic for production)
        if duration == 0:
            return None, None, None

        # Calculate time points for frame sampling
        frame_timestamps = np.linspace(0, duration, num_frames, endpoint=False)

        frame_embeddings = []

        # Use the CPU for image processing and feature extraction
        device = "cpu"  # For a Streamlit app without easy GPU access
        clip_model.to(device)

        for t in frame_timestamps:
            frame_pil = Image.fromarray(clip.get_frame(t))
            inputs = clip_processor(images=frame_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                # Get the embedding for the frame
                frame_features = clip_model.get_image_features(**inputs)

            # Normalize and append
            frame_embeddings.append(frame_features / frame_features.norm(dim=-1, keepdim=True))

        clip.close()  # Important to close the clip object to release resources

        # (N_frames, 1, D) -> (N_frames, D)
        frame_embeddings_tensor = torch.stack(frame_embeddings).squeeze(1)

        # Global Video Feature (for overall relevance)
        global_embedding = frame_embeddings_tensor.mean(dim=0)

        # Move back to CPU numpy for storage
        return global_embedding.cpu().numpy(), frame_embeddings_tensor.cpu().numpy(), frame_timestamps

    except Exception as e:
        # In a parallel environment, print or log the error
        print(f"Error processing video {os.path.basename(video_path)}: {e}")
        return None, None, None


def get_text_embedding(text):
    """Computes the CLIP embedding for a text query."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    # Normalize and convert to numpy
    return (text_features / text_features.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()


def find_best_moment(text_embedding, frame_embeddings, timestamps, moment_duration=5.0):
    """
    Simulates X-CLIP's local alignment by finding the frame most similar to the query.
    """
    # frame_embeddings shape: (N_frames, D)
    # text_embedding shape: (D,)

    # 1. Calculate similarity score for every frame
    frame_similarities = np.dot(frame_embeddings, text_embedding)

    # 2. Find the frame with the highest similarity
    best_frame_index = np.argmax(frame_similarities)
    max_similarity = frame_similarities[best_frame_index]

    # 3. Determine the 5-second moment around that frame
    best_time = timestamps[best_frame_index]

    # Calculate half the moment duration
    half_moment = moment_duration / 2

    last_frame_duration = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else moment_duration
    video_end_time = timestamps[-1] + last_frame_duration

    start_time = max(0.0, best_time - half_moment)
    end_time = min(video_end_time, best_time + half_moment)

    return start_time, end_time, max_similarity


def process_video_for_db(full_path, original_name):
    """
    A wrapper to process a video and return its data for the database.
    This is run in a separate thread.
    """
    try:
        # For simplicity with the existing structure, we need a 'path' even for /data files.
        # However, for /data files, we can just use their full_path directly, no need for tempfile.
        # If running in a system where /data is volatile, you might copy it to a temp dir.
        # For this scenario, we'll use the original full_path.
        video_path_to_use = full_path

        global_embed, frame_embeds, timestamps = get_video_embeddings_and_temporal_info(
            video_path_to_use, num_frames=FRAME_COUNT
        )

        if global_embed is not None:
            return {
                'name': original_name,
                'path': video_path_to_use,
                'global_embedding': global_embed,
                'frame_embeddings': frame_embeds,
                'timestamps': timestamps
            }
        else:
            return None

    except Exception as e:
        print(f"Failed to process video {original_name}: {e}")
        return None


def ingest_data_folder():
    """Loads all MP4 files from the DATA_FOLDER in parallel."""
    if not os.path.isdir(DATA_FOLDER):
        st.error(f"Error: The directory '{DATA_FOLDER}' was not found. Cannot ingest data.")
        return

    video_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".mp4")]

    if not video_files:
        st.warning(f"No .mp4 files found in the '{DATA_FOLDER}' directory.")
        return

    st.info(f"Found {len(video_files)} video(s) to index in '{DATA_FOLDER}'. This may take a moment...")

    # Use ThreadPoolExecutor for concurrent I/O-bound tasks (moviepy reading)
    # Be mindful of memory consumption and thread limits.
    max_workers = min(os.cpu_count() * 2, len(video_files)) if os.cpu_count() else 4
    results = []

    with st.spinner(f"Indexing {len(video_files)} video(s) using up to {max_workers} threads..."):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing tasks
            futures = {
                executor.submit(
                    process_video_for_db,
                    os.path.join(DATA_FOLDER, filename),
                    filename  # Use the filename as the key
                ): filename
                for filename in video_files
                if filename not in st.session_state.video_db  # Skip already indexed videos
            }

            total_tasks = len(futures)
            if total_tasks == 0:
                st.info("All found videos are already indexed.")
                return

            progress_bar = st.progress(0, text="Indexing progress...")
            processed_count = 0

            for future in as_completed(futures):
                result = future.result()
                processed_count += 1
                progress_bar.progress(processed_count / total_tasks,
                                      text=f"Indexing progress: {processed_count}/{total_tasks}")

                if result:
                    st.session_state.video_db[result['name']] = {
                        'path': result['path'],
                        'global_embedding': result['global_embedding'],
                        'frame_embeddings': result['frame_embeddings'],
                        'timestamps': result['timestamps']
                    }
                    results.append(result['name'])

            progress_bar.empty()  # Remove the progress bar
            st.session_state.video_ids = list(st.session_state.video_db.keys())

    if results:
        st.success(f"Successfully indexed {len(results)} new video(s) from '{DATA_FOLDER}'.")
    else:
        st.warning("No new videos were indexed (either none found or errors occurred).")


# --- Streamlit UI ---
st.title("Video-Language semantic search")
st.markdown(f"Using **{MODEL_NAME}** for feature extraction and **Frame-to-Text alignment** for temporal matching.")
st.markdown("---")

# =========================
# Sidebar: Data Management
# =========================
with st.sidebar:
    st.header("📦 Data Manager")
    st.caption(f"Data folder: `{DATA_FOLDER}`")

    # Ingest all videos from folder
    st.subheader("Ingest from folder")
    st.markdown("Load all **.mp4** videos found in the data directory.")
    if st.button("🚀 Ingest Data", key="ingest_button", use_container_width=True):
        ingest_data_folder()

    st.divider()

    # Upload a single video
    st.subheader("Upload a video")
    uploaded_file = st.file_uploader("Upload an MP4", type="mp4", key="uploader")

    if uploaded_file is not None:
        video_name = uploaded_file.name

        if video_name not in st.session_state.video_db:
            progress_bar = st.progress(0, text=f"Processing {video_name}...")

            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            # Store the temp path so we can try to clean it up later if needed
            st.session_state.temp_files[video_name] = temp_path

            # 1. Get Embeddings
            global_embed, frame_embeds, timestamps = get_video_embeddings_and_temporal_info(
                temp_path, num_frames=FRAME_COUNT
            )

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
                st.error(f"Failed to index video '{video_name}'.")
                os.unlink(temp_path)  # Clean up temp file on error or failure
                if video_name in st.session_state.temp_files:
                    del st.session_state.temp_files[video_name]  # Remove from temp file list

    st.divider()

    # Database status lives in sidebar
    if st.session_state.video_db:
        st.success(f"Indexed videos: **{len(st.session_state.video_db)}**")
    else:
        st.warning("Database is empty. Upload or ingest a video to begin.")


st.header("Semantic Search and Moment Retrieval")

query_text = st.text_input(
    "Enter your text query:",
    placeholder="e.g., A slow-motion shot of a cat jumping"
)

# Search button is disabled if the database is empty
search_button = st.button("Search Videos & Find Moments", type="primary", disabled=not st.session_state.video_db)

if search_button and query_text:

    # 1. Get text query embedding (Global Text Feature)
    with st.spinner(f"Encoding query: '{query_text}'..."):
        query_embedding = get_text_embedding(query_text)

    st.subheader(f"Results for: **'{query_text}'**")

    # List to hold video scores and details
    video_scores = []

    # 2. Score All Videos (Global Retrieval and Local Grounding)
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
            # Use 'dot' product as a proxy for cosine similarity
            st.markdown(f"**Overall Relevance (Global):** {result['global_similarity']:.4f}")
            st.markdown(f"**Best Moment Relevance (Local):** {result['max_local_similarity']:.4f}")

            # Show the best 5-second window
            st.markdown(f"**📺 Best Moment Found:** `{result['start_time']:.1f}s` to `{result['end_time']:.1f}s`")

        with col2:
            st.video(
                result['path'],
                start_time=int(result['start_time']),
                # Note: Setting end_time for the Streamlit video player is not directly supported,
                # but the start_time jumps the user to the relevant part.
            )
            st.caption(f"Video starts at the most relevant moment ({result['start_time']:.1f}s).")

        st.markdown("---")

# Note: In a real-world scenario, you'd add a st.balloons() or st.success() after ingestion,
# and more robust error handling and temporary file cleanup.