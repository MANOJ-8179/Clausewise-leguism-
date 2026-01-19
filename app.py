import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import base64
import io

# Function to set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local("img.avif")

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to summarize text
def summarize(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to convert text to speech using gTTS
def gtts_text_to_speech(text):
    try:
        tts = gTTS(text, lang='en')
        audio_buffer = io.BytesIO()
        tts.save("output_audio.mp3")
        with open("output_audio.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        return io.BytesIO(audio_bytes)
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

# Streamlit app interface
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FFFFFF;">ClauseWise Legisum</h1>
        <h3 style="color: #FFFFFF;">Summarize your terms and conditions with ease!</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Text input field for user to input the Terms & Conditions text
text_input = st.text_area("Paste your Terms and Conditions here:", height=300)

# Button to generate the summary
if st.button("✨ Generate Summary"):
    if text_input:  # Ensure the input is not empty
        with st.spinner("Generating summary..."):
            summary = summarize(text_input)
            st.markdown(f"<p style='font-size:20px;'>{summary}</p>", unsafe_allow_html=True)

            # Convert text to speech using gTTS
            audio_stream = gtts_text_to_speech(summary)

            if audio_stream:
                audio_base64 = base64.b64encode(audio_stream.read()).decode()
                st.markdown(
                    f"""
                    <audio autoplay controls>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("Please paste the Terms and Conditions text first.")
