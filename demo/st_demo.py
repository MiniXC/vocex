import io

import streamlit as st
import numpy as np
from vocex import Vocex
import time
from audio_recorder_streamlit import audio_recorder
from datasets import load_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import librosa
import torchaudio
from scipy.io.wavfile import write
import pandas as pd

# set page title
st.set_page_config(page_title='Vocex Demo', page_icon='🎤')

st.title('Vocex Demo')

audio_data = None
audio_data_arr = None


def set_file_changed():
    st.session_state['file_changed'] = True

def clear_audio():
    global audio_data, audio_data_arr, sample_rate
    st.session_state['audio_data'] = None
    st.session_state['audio_data_arr'] = None
    st.session_state['sample_rate'] = None
    st.session_state['output'] = None
    audio_data = None
    audio_data_arr = None
    sample_rate = None

with st.sidebar:
    checkpoint = st.selectbox('Model Checkpoint', ['cdminix/vocex', 'models/checkpoint_full.ckpt', 'models/checkpoint_half.ckpt'], index=0)

    if 'checkpoint' in st.session_state and st.session_state['checkpoint'] != checkpoint:
        clear_audio()
        st.session_state['checkpoint'] = checkpoint
        st.session_state['model'] = None
    
    st.session_state['checkpoint'] = checkpoint

    # load model
    if 'model' not in st.session_state:
        with st.spinner('Loading model...'):
            st.session_state['model'] = Vocex.from_pretrained(checkpoint)
            model = st.session_state['model']
    else:
        model = st.session_state['model']

    cmap = st.selectbox('Spectrogram Color Map', ['magma', 'viridis', 'plasma', 'inferno', 'gray'], index=4)

with st.expander('Select Audio Data'):

    src = st.radio('Audio Source', ['Upload Audio File', 'Record Audio', 'Random CommonVoice Audio'], index=2)

    if src == 'Upload Audio File':
        audio_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3', 'ogg', 'flac'], on_change=set_file_changed)
        if audio_file is not None:
            audio_data_new = audio_file.read()
            if 'file_changed' not in st.session_state:
                st.session_state['file_changed'] = False
            file_changed = st.session_state['file_changed']
            if audio_data_new is not None and file_changed: # only clear if file changed
                clear_audio()
                st.session_state['audio_data'] = audio_data_new
                audio_data = audio_data_new
                st.session_state['file_changed'] = False

    if src == 'Record Audio':
        audio_data_new = audio_recorder()
        if audio_data_new is not None and len(audio_data_new) > 1_000:
            clear_audio()
            st.session_state['audio_data'] = audio_data_new
            audio_data = audio_data_new
        if audio_data_new is not None and len(audio_data_new) <= 1_000:
            st.warning('Audio recording too short, please try again')

    if src == 'Random CommonVoice Audio' or 'audio_data_arr' not in st.session_state:
        if st.button('Get Random Audio') or 'audio_data_arr' not in st.session_state:
            with st.spinner('Loading data...'):
                example_num = np.random.randint(100)
                # make sure this is different every time
                example_num = (example_num + int(time.time())) % 100
                
                audio_file = f'demo/100_examples/{example_num}.wav'
            
                clear_audio()
                audio_data_arr, sample_rate = librosa.load(audio_file, sr=None)
                st.session_state['audio_data_arr'] = audio_data_arr
                st.session_state['sample_rate'] = sample_rate

if 'audio_data_arr' in st.session_state:
    audio_data_arr = st.session_state['audio_data_arr']

if 'audio_data' in st.session_state:
    audio_data = st.session_state['audio_data']

if 'sample_rate' in st.session_state:
    sample_rate = st.session_state['sample_rate']

if audio_data is not None or audio_data_arr is not None:
    st.subheader('Selected Audio')
    st.button('Clear Audio', key='clear_audio', on_click=clear_audio)
    if audio_data_arr is None:
        # load from audio_data bytes
        # create "fake file" to load into librosa
        audio_data_arr, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None)
        st.session_state['audio_data_arr'] = audio_data_arr
        st.session_state['sample_rate'] = sample_rate
    st.audio(audio_data_arr, sample_rate=sample_rate, format='audio/ogg')

if audio_data_arr is not None:
    # inference
    with st.spinner('Inference...'):
        start = time.time()
        if sample_rate is None:
            st.write('Please select sample rate')
        else:
            if 'output' not in st.session_state or st.session_state['output'] is None:
                output = model(audio_data_arr, sr=sample_rate, return_activations=True, return_attention=True, speaker_avatar=True)
                end = time.time()
                st.write(f'Inference time: {end - start:.2f}s')
                st.session_state['output'] = output
                if 'measures' not in st.session_state:
                    st.session_state['measures'] = list(output["measures"].keys())
            output = st.session_state['output']
            measure_keys = st.session_state['measures']
            all_measures = st.checkbox('Show all measures', value=True, key='show_all_measures')
            if not all_measures:
                measure = st.selectbox('Measure', measure_keys, index=1)
            fig = plt.figure(figsize=(20, 6))
            mel = model._preprocess(audio_data_arr, sample_rate)[0]
            plt.imshow(mel.T, aspect="auto", origin="lower", cmap=cmap, alpha=0.5)
            plt.twinx()
            if all_measures:
                for m in reversed(measure_keys):
                    if 'voice' not in m:
                        values = np.array(output["measures"][m])[0]
                        values = (values - values.min()) / (values.max() - values.min())
                        # get sns color palette, and make line a bit thicker
                        palette = sns.color_palette()
                        measure_index = measure_keys.index(m)
                        values += measure_index * 1.3
                        sns.lineplot(x=range(len(values)), y=values, color=palette[measure_index], linewidth=5)
                        plt.text(-70, measure_index * 1.3 + .5, m, fontsize=40, color=palette[measure_index])
                        plt.title("all measures")
                        plt.tight_layout()
            else:
                values = np.array(output["measures"][measure])[0]
                # get sns color palette, and make line a bit thicker
                palette = sns.color_palette()
                measure_index = measure_keys.index(measure)
                sns.lineplot(x=range(len(values)), y=values, color=palette[measure_index], linewidth=5)
                plt.title(measure)
                plt.tight_layout()
            plt.xlim(0, len(values))
            if all_measures:
                plt.ylim(0, 5)
            else:
                val_range = values.max() - values.min()
                plt.ylim(values.min() - val_range * .1, values.max() + val_range * .1)
            st.pyplot(fig)
            st.subheader('Overall Measures')
            st.table(pd.DataFrame([[np.round(float(output[f'overall_{m}']), 2) for m in measure_keys if m != 'voice_activity_binary']], columns=[m for m in measure_keys if m != 'voice_activity_binary'], index=['value']))
            st.subheader("Advanced")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('D-Vector')
                # write as code block
                st.code(output['dvector'][0].tolist())
                st.write("unique generated image")
                st.image(output['avatars'][0], width=200)
            with col2:
                st.subheader('Attention')
                layer = st.selectbox('Attention Layer', list(range(output['attention'].shape[1])), index=0)
                attn = output['attention'][0][layer]
                # min-max normalize
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                if st.checkbox('Log Scale', value=True, key='log_scale_attn'):
                    attn = np.log(attn + 1e-5)
                    attn = (attn - attn.min()) / (attn.max() - attn.min())
                st.image(attn, width=200)
            with col3:
                st.subheader('Activations')
                layer = st.selectbox('Activation Layer', list(range(output['activations'].shape[1])), index=0)
                act = output['activations'][0][layer]
                # min-max normalize
                act = (act - act.min()) / (act.max() - act.min())
                if st.checkbox('Log Scale', value=False, key='log_scale_act'):
                    act = np.log(act + 1e-5)
                    act = (act - act.min()) / (act.max() - act.min())
                st.image(act.T, width=200)