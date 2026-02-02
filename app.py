import streamlit as st
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import io
import os
import zipfile

def create_audio_segment(base_freq, offset, sample_rate, duration, include_ref):
    """Helper to generate a single AudioSegment."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float64)
    freq = np.float64(base_freq) + np.float64(offset)
    
    # Generate signals
    sig = np.sin(2 * np.pi * freq * t)
    
    if include_ref:
        ref = np.sin(2 * np.pi * base_freq * t)
        final = np.vstack((sig, ref)).T
    else:
        final = np.vstack((sig, sig)).T

    # Convert to 16-bit PCM
    audio_data = (final * 32767).astype(np.int16)
    
    wav_buf = io.BytesIO()
    wavfile.write(wav_buf, sample_rate, audio_data)
    wav_buf.seek(0)
    return AudioSegment.from_wav(wav_buf)

def export_buffer(audio_segment, fmt):
    """Converts AudioSegment to desired format buffer."""
    buf = io.BytesIO()
    audio_segment.export(buf, format=fmt.lower())
    buf.seek(0)
    return buf

# --- Streamlit UI ---
st.set_page_config(page_title="Precision Audio Lab", page_icon="🔊")
st.title("🔊 Precision Audio Generator")

with st.sidebar:
    st.header("1. Global Settings")
    base_freq = st.number_input("Base Frequency (Hz)", value=440.0, step=1.0, format="%.1f")
    sample_rate = st.selectbox("Sample Rate (Hz)", [44100, 48000, 88200, 96000], index=3)
    duration = st.slider("Duration (Seconds)", 1, 120, 60)
    
    st.header("2. Generation Mode")
    gen_mode = st.radio("Output Style", ["Concurrent Sweep (Single File)", "Individual Files (ZIP)"])
    
    st.header("3. Export Options")
    file_format = st.selectbox("Format", ["wav", "mp3", "flac", "opus"])
    include_ref = st.checkbox("Include Reference Channel", value=True)

# Offset Inputs
col1, col2 = st.columns(2)
with col1:
    decimals = st.slider("Precision Depth (10^-x)", 1, 15, 5)
    offsets1 = [10**-i for i in range(1, decimals + 1)]
with col2:
    raw_ints = st.text_input("Integer offsets (comma separated)", "1, 10, 100")
    offsets2 = [float(x.strip()) for x in raw_ints.split(",") if x.strip()]

all_offsets = offsets1 + offsets2

if st.button("Generate Audio", type="primary"):
    with st.spinner("Processing..."):
        if gen_mode == "Concurrent Sweep (Single File)":
            # MIXED LOGIC
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float64)
            mixed = np.zeros_like(t)
            for off in all_offsets:
                mixed += np.sin(2 * np.pi * (base_freq + off) * t)
            
            # Normalization
            mixed /= np.max(np.abs(mixed))
            
            # Create AudioSegment
            if include_ref:
                ref = np.sin(2 * np.pi * base_freq * t)
                final = np.vstack((mixed, ref)).T
            else:
                final = np.vstack((mixed, mixed)).T
                
            audio_data = (final * 32767).astype(np.int16)
            wav_buf = io.BytesIO()
            wavfile.write(wav_buf, sample_rate, audio_data)
            wav_buf.seek(0)
            
            segment = AudioSegment.from_wav(wav_buf)
            out_file = export_buffer(segment, file_format)
            
            st.audio(out_file, format=f"audio/{file_format}")
            st.download_button(f"📥 Download Sweep ({file_format.upper()})", out_file, f"sweep_{base_freq}Hz.{file_format}")

        else:
            # INDIVIDUAL FILES LOGIC
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for i, off in enumerate(all_offsets):
                    segment = create_audio_segment(base_freq, off, sample_rate, duration, include_ref)
                    buf = export_buffer(segment, file_format)
                    
                    # Naming convention: SeqNo_BaseFreq_Offset_SR.fmt
                    clean_off = f"{off:.15f}".rstrip('0').rstrip('.')
                    fname = f"{i+1}_{base_freq}Hz_plus_{clean_off}Hz_{sample_rate}SR.{file_format}"
                    zip_file.writestr(fname, buf.getvalue())
            
            zip_buffer.seek(0)
            st.success(f"Generated {len(all_offsets)} files!")
            st.download_button("📥 Download All Files (ZIP)", zip_buffer, "audio_batch.zip", "application/zip")
