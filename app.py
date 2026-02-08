import streamlit as st
import pandas as pd
import time
import torch
import json
import subprocess
import os
import sys
from PIL import Image
from task import Net, get_transforms, get_num_classes

STATE_FILE = "simulation_state.json"

# Dataset Configuration for Inference
DATASET_CONFIG = {
    "chest_xray": {
        "label": "Chest X-Ray (Pneumonia)",
        "classes": ["NORMAL", "PNEUMONIA"],
        "num_classes": 2
    },
    "Alzheimer": {
        "label": "Alzheimer's MRI",
        "classes": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
        "num_classes": 4
    }
}

st.set_page_config(page_title="FL Healthcare", layout="wide")

st.title("Federated Learning for Smart Healthcare")
st.markdown("### Privacy-Preserving Medical Imaging Analysis")

# --- Sidebar ---
st.sidebar.header("Configuration")
selected_dataset_key = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASET_CONFIG.keys()),
    format_func=lambda x: DATASET_CONFIG[x]["label"]
)

num_rounds = st.sidebar.slider("Number of FL Rounds", min_value=1, max_value=10, value=3)
start_btn = st.sidebar.button("Start Federated Training")

# --- Main Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Training Progress ({DATASET_CONFIG[selected_dataset_key]['label']})")
    chart_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.empty()

with col2:
    st.subheader("Inference")
    uploaded_file = st.file_uploader("Upload Image for Prediction", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict"):
            try:
                # Determine model file and config
                dataset_conf = DATASET_CONFIG[selected_dataset_key]
                model_path = f"global_model_{selected_dataset_key}.pth"
                
                # Check if specific model exists
                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"Model file {model_path} not found. Train first.")

                # Load Model
                model = Net(num_classes=dataset_conf["num_classes"])
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                # Preprocess
                transform = get_transforms()
                img_t = transform(image).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = model(img_t)
                    _, predicted = torch.max(output, 1)
                    prop = torch.nn.functional.softmax(output, dim=1)
                
                res = dataset_conf["classes"][predicted.item()]
                confidence = prop[0][predicted.item()].item()
                
                st.success(f"Prediction: **{res}**")
                st.info(f"Confidence: {confidence:.2f}")
                
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# --- Simulation Logic (Decoupled Process) ---

if "history" not in st.session_state:
    st.session_state["history"] = []
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# Display logs
st.sidebar.divider()
st.sidebar.subheader("🔒 Privacy Logs")
log_placeholder = st.sidebar.empty()

def render_logs():
    with log_placeholder.container():
        # st.sidebar.subheader(" Privacy Logs") 
        for log in st.session_state["logs"]:
            st.text(f"> {log}")

render_logs()

if start_btn:
    st.session_state["history"] = []
    st.session_state["logs"] = []
    # Clear logs in UI
    log_placeholder.empty()
    
    status_text.text("Status: Initializing background process...")
    progress_bar.progress(0)
    
    # Clean up old state file
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except:
            pass

    # Launch background process
    # Use CREATE_BREAKAWAY_FROM_JOB (0x01000000) to detach from the parent Job Object
    creation_flags = 0x01000000 # subprocess.CREATE_BREAKAWAY_FROM_JOB
    
    cmd = [sys.executable, "backend_runner.py", "--rounds", str(num_rounds), "--dataset", selected_dataset_key]
    
    # Try with BREAKAWAY; if strictly restricted, might need CREATE_NEW_CONSOLE (0x00000010)
    process = subprocess.Popen(cmd, creationflags=creation_flags)
    
    st.info(f"Launched simulation process (PID: {process.pid}) for {selected_dataset_key}")
    
    # Polling loop
    while True:
        if process.poll() is not None:
             # Process finished
             break
        
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                
                # Update UI
                current_round = data.get("round", 0)
                acc = data.get("accuracy", 0.0)
                logs = data.get("logs", [])
                status = data.get("status", "RUNNING")
                
                # Update logs
                if len(logs) > len(st.session_state["logs"]):
                    st.session_state["logs"] = logs
                    render_logs()

                # Update chart
                existing_rounds = [item["Round"] for item in st.session_state["history"]]
                if current_round > 0 and current_round not in existing_rounds:
                     st.session_state["history"].append({"Round": current_round, "Accuracy": acc})
                
                if st.session_state["history"]:
                    df = pd.DataFrame(st.session_state["history"])
                    chart_placeholder.line_chart(df.set_index("Round"))
                
                status_text.text(f"Status: {status} | Round {current_round}/{num_rounds} | Accuracy: {acc:.4f}")
                progress_bar.progress(min(current_round / num_rounds, 1.0))
                
                if status in ["COMPLETED", "FAILED"]:
                    break
                    
            except json.JSONDecodeError:
                pass # partial write
            except Exception as e:
                pass 
        
        time.sleep(1)

    if process.returncode == 0:
        status_text.success("Training Complete!")
        progress_bar.progress(100)
    else:
        status_text.error(f"Simulation process failed (Code {process.returncode}). Check terminal output.")

