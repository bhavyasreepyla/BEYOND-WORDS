import streamlit as st
import pyttsx3

def text_to_speech(text, rate=150, volume=1.0, voice_id=None):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # properties
    engine.setProperty('rate', rate)  # Speed of speech
    engine.setProperty('volume', volume)  # Volume level (0.0 to 1.0)

    voices = engine.getProperty('voices')

    if voice_id is not None:
        engine.setProperty('voice', voices[voice_id].id)

    # Convert the text to speech
    engine.say(text)

    engine.runAndWait()

def main():
    st.title("Text-to-Speech Streamlit App")

    # user input
    text_input = st.text_area("Enter the text you want to convert to speech:")

    # speech speed
    speech_speed = st.slider("Select speech speed", min_value=50, max_value=300, value=150)

    # volume
    volume_level = st.slider("Select volume level", min_value=0.0, max_value=1.0, value=1.0)

    # voice selection
    voices = pyttsx3.init().getProperty('voices')
    voice_choices = [voice.name for voice in voices]
    voice_choice = st.selectbox("Select a voice", voice_choices, index=0)

    if st.button("Convert to Speech"):
        
        voice_id = voice_choices.index(voice_choice)
        
        # Call the function to convert text to speech with user choices
        text_to_speech(text_input, speech_speed, volume_level, voice_id)
        st.success("Speech generated successfully!")

if __name__ == "__main__":
    main()
