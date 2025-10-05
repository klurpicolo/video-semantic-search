import streamlit as st
import torch, faiss, numpy as np
from transformers import XCLIPProcessor, XCLIPModel
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
    model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model.eval()
    return model, processor

def sample_frames(video_bytes, num_frames=16):
    # write bytes to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video with OpenCV")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or num_frames
    idxs = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        # BGR -> RGB, then to PIL.Image to keep your downstream pipeline
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from video")
    return frames


@torch.inference_mode()
def video_embedding(model, processor, frames):
    inputs = processor(videos=[frames], return_tensors="pt")
    out = model(**inputs)
    # X-CLIP returns pooled embeddings
    vid_emb = out.video_embeds[0]           # (d,)
    vid_emb = torch.nn.functional.normalize(vid_emb, dim=0)
    return vid_emb.cpu().numpy()

@torch.inference_mode()
def text_embedding(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt")
    out = model.get_text_features(**inputs)  # or model(**inputs).text_embeds
    txt = torch.nn.functional.normalize(out[0], dim=0)
    return txt.cpu().numpy()

st.title("Video semantic search (X-CLIP)")
model, processor = load_model()

# 1) Upload/build index
st.header("1) Add videos")
uploaded = st.file_uploader("Upload MP4 videos", type=["mp4"], accept_multiple_files=True)

# Simple in-memory store
if "vid_titles" not in st.session_state:
    st.session_state.vid_titles, st.session_state.vecs = [], None

# if uploaded:
#     vecs = []
#     titles = []
#     for f in uploaded:
#         frames = sample_frames(f.read(), num_frames=16)
#         emb = video_embedding(model, processor, frames)
#         vecs.append(emb)
#         titles.append(f.name)
#     V = np.stack(vecs, axis=0).astype("float32")
#     st.session_state.vecs = V if st.session_state.vecs is None else np.vstack([st.session_state.vecs, V])
#     st.session_state.vid_titles.extend(titles)
#     st.success(f"Indexed {len(uploaded)} video(s). Total: {len(st.session_state.vid_titles)}")

# In your Streamlit app, replace the video processing loop:
if uploaded:
    vecs = []
    titles = []
    for f in uploaded:
        try:
            # f.read() can only be called once, so read it outside the try/except
            video_bytes = f.read()
            frames = sample_frames(video_bytes, num_frames=16)
            emb = video_embedding(model, processor, frames)
            vecs.append(emb)
            titles.append(f.name)
        except RuntimeError as e:
            # Catch the "No frames decoded" error from your function
            st.error(
                f"Failed to process video '{f.name}': {e}. This often means OpenCV can't decode the video's codec.")
            continue  # Skip to the next file
        except Exception as e:
            # Catch other potential errors during embedding
            st.error(f"An unexpected error occurred while processing '{f.name}': {e}")
            continue

    if vecs:  # Only update session state if some videos were successfully processed
        V = np.stack(vecs, axis=0).astype("float32")
        # Ensure st.session_state.vecs is initialized as an array if it's the first upload
        if st.session_state.vecs is None or st.session_state.vecs.size == 0:
            st.session_state.vecs = V
        else:
            st.session_state.vecs = np.vstack([st.session_state.vecs, V])

        st.session_state.vid_titles.extend(titles)
        st.success(f"Indexed {len(vecs)} video(s). Total: {len(st.session_state.vid_titles)}")
    else:
        st.warning("No videos were successfully indexed.")

# Build FAISS index when we have vectors
index = None
if st.session_state.vecs is not None:
    dim = st.session_state.vecs.shape[1]
    index = faiss.IndexFlatIP(dim)                 # cosine since we normalized
    index.add(st.session_state.vecs)

# 2) Query
st.header("2) Search")
q = st.text_input("Describe what you want (e.g., 'a person riding a bicycle on a beach')")
k = st.slider("Top-K", 1, 10, 5)

if st.button("Search") and q and index is not None:
    qvec = text_embedding(model, processor, q).astype("float32")[None, :]
    sims, idxs = index.search(qvec, k)
    st.subheader("Results")
    for rank, (i, s) in enumerate(zip(idxs[0], sims[0]), start=1):
        st.write(f"#{rank}: {st.session_state.vid_titles[i]}  (score: {float(s):.3f})")
