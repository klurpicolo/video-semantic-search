import streamlit as st
import torch
import numpy as np
import tempfile
import moviepy as mp
import os
from PIL import Image


# --- Configuration for X-CLIP Model ---
# We use a base X-CLIP model that is available on Hugging Face
MODEL_NAME = "microsoft/xclip-base-patch16"


# --- Initialization ---

@st.cache_resource
def load_xclip_model():
    """Loads the X-CLIP model and processor only once."""
    try:
        from transformers import XCLIPProcessor, XCLIPModel
        st.write(f"Loading X-CLIP model: {MODEL_NAME}...")

        # Load the model and its processor (which handles video and text inputs)
        model = XCLIPModel.from_pretrained(MODEL_NAME)
        processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
        model.eval()
        return model, processor
    except ImportError:
        st.error("Error: The XCLIP classes were not found. Please ensure you have the latest 'transformers' library.")
        return None, None


# Load model globally
xclip_model, xclip_processor = load_xclip_model()

if not xclip_model:
    st.stop()

# Global state for the vector database
if 'video_db' not in st.session_state:
    st.session_state.video_db = {}
    st.session_state.video_ids = []


# --- Core Video Processing and Retrieval ---

def extract_video_frames(video_path, num_frames=8):
    """
    Extracts a list of PIL Images from the video. The number of frames
    is defined by the model's processor config (typically 8-16 for X-CLIP).
    This function replaces the manual frame sampling loop in the previous code.
    """
    try:
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration

        # Determine sampling times
        frame_timestamps = np.linspace(0, duration, num_frames, endpoint=False)

        frames = [
            Image.fromarray(clip.get_frame(t))
            for t in frame_timestamps
        ]

        return frames, frame_timestamps, duration
    except Exception as e:
        st.error(f"Error extracting frames from video: {e}")
        return [], [], 0


def get_video_and_text_features(video_frames, query_text):
    """
    Computes both the video and text embeddings using the X-CLIP processor/model.

    NOTE: When calling get_video_features, we only need to pass 'pixel_values'.
    When calling get_text_features, we only need to pass 'input_ids' and 'attention_mask'.
    """
    # 1. Process inputs for both modalities
    inputs = xclip_processor(
        videos=video_frames,
        text=[query_text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        # 2. Get Video Features (Corrected Call)
        # We pass only the required positional argument 'pixel_values'
        outputs_video_f = xclip_model.get_video_features(
            pixel_values=inputs.pixel_values
        )

        # 3. Get Text Features (Corrected Call)
        # We pass only the text arguments
        outputs_text_f = xclip_model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        # Normalize and convert to NumPy
        video_features = outputs_video_f / outputs_video_f.norm(dim=-1, keepdim=True)
        text_features = outputs_text_f / outputs_text_f.norm(dim=-1, keepdim=True)

        return video_features.squeeze().cpu().numpy(), text_features.squeeze().cpu().numpy()


def find_best_moment_xclip(video_path, query_text, duration, max_clips=10):
    """
    Performs a simplified temporal grounding by slicing the video and scoring sub-clips.
    This simulates the moment retrieval capability.
    """
    # XCLIP's actual temporal grounding is complex, so we simulate it by scoring
    # overlapping short clips of the video.

    clip_duration = 5.0  # e.g., 5-second window
    step_size = 2.0  # Slide the window by 2 seconds

    best_similarity = -1.0
    best_moment = (0.0, clip_duration)

    # Generate overlapping 5-second time windows
    timestamps = np.arange(0.0, duration - step_size, step_size)

    for start_t in timestamps:
        end_t = min(start_t + clip_duration, duration)

        # Skip very short trailing segments
        if end_t - start_t < step_size:
            continue

            # 1. Extract frames for the *sub-clip*
        sub_clip_frames = []
        try:
            clip = mp.VideoFileClip(video_path)
            # Sample 8 frames within this sub-clip (the model's expectation)
            frame_times = np.linspace(start_t, end_t, 8, endpoint=False)
            sub_clip_frames = [Image.fromarray(clip.get_frame(t)) for t in frame_times]
        except Exception:
            continue

        if not sub_clip_frames:
            continue

        # 2. Get the sub-clip's embedding and the text embedding
        # We reuse the model's logic for the text feature, but only compute once for efficiency

        # We only need the video feature for the sub-clip
        video_inputs = xclip_processor(videos=sub_clip_frames, return_tensors="pt")
        text_inputs = xclip_processor(text=[query_text], return_tensors="pt")

        with torch.no_grad():
            video_features_sub = xclip_model.get_video_features(video_inputs.pixel_values)  # <-- Already correct!
            text_features_sub = xclip_model.get_text_features(text_inputs.input_ids, text_inputs.attention_mask)

            # Normalize
            video_features_sub = video_features_sub / video_features_sub.norm(dim=-1, keepdim=True)
            text_features_sub = text_features_sub / text_features_sub.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = torch.dot(video_features_sub.squeeze(), text_features_sub.squeeze()).item()

        if similarity > best_similarity:
            best_similarity = similarity
            best_moment = (start_t, end_t)

    return best_moment, best_similarity


# --- Streamlit UI ---

st.title("🎬 X-CLIP Semantic Search & Temporal Pinpointing")
st.markdown("Retrieval is powered by the Hugging Face **`microsoft/xclip-base-patch16-v5`** model.")
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

        # Extract frames and compute global features (using a dummy query for initial feature extraction)
        video_frames, _, duration = extract_video_frames(temp_path)

        if video_frames:
            # We use a dummy text to get the global feature, as the model expects both inputs
            global_embed, _ = get_video_and_text_features(video_frames, "a video")

            # Update database
            st.session_state.video_db[video_name] = {
                'path': temp_path,
                'global_embedding': global_embed,
                'duration': duration
            }
            st.session_state.video_ids = list(st.session_state.video_db.keys())

            progress_bar.progress(100, text=f"Processing {video_name}... Done!")
            st.success(f"Video '{video_name}' added to the database!")
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
    "Enter your search query:",
    placeholder="e.g., A dog running across a field"
)

search_button = st.button("Search Videos & Find Moments", type="primary", disabled=not st.session_state.video_db)

if search_button and query_text:

    # 1. Get query embedding
    with st.spinner(f"Encoding query: '{query_text}'..."):
        text_embedding = xclip_model.get_text_features(
            **xclip_processor(text=[query_text], return_tensors="pt", padding=True)
        )
        text_embedding = text_embedding.squeeze().cpu().numpy()

    st.subheader(f"Top Results for: **'{query_text}'**")
    video_scores = []

    # 2. Score All Videos (Global Retrieval)
    for video_name, data in st.session_state.video_db.items():
        # Global Similarity (Video-level match)
        global_similarity = np.dot(data['global_embedding'], text_embedding)

        # 3. Temporal Grounding (Local Match - Heavy Computation)
        with st.spinner(f"Finding best moment in {video_name}..."):
            (start_t, end_t), max_local_sim = find_best_moment_xclip(
                data['path'],
                query_text,
                data['duration']
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

    # 4. Display Results
    for rank, result in enumerate(ranked_videos[:3]):  # Show top 3 videos

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
                start_time=int(result['start_time'])
            )
            st.caption("Video playback starts at the pinned moment.")

        st.markdown("---")