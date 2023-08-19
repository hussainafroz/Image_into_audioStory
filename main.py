import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub,LLMChain
from transformers import pipeline
from huggingface_hub import hf_hub_download
import streamlit as st
 
load_dotenv()
TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ### IMG TO TEXT
#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-large") #pipeline(task , model)
    text = image_to_text(
         url)[0]['generated_text'] #['generated_text'] #only to take the text generated
    print(text)
    return text
 
# ###  LLM (TEXT TO STORY TEXT)
 
##llm

from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
# input_prompt = scenario
# story = generator(input_prompt, max_length=80, do_sample=True,repetition_penalty=1.5, temperature=1.2, 
#                top_p=0.95, top_k=50)
def generate_story(scenario):
      
        model_name = "aspis/gpt2-genre-story-generation"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        input_prompt = scenario
        story = generator(input_prompt, max_length=60, do_sample=True,repetition_penalty=1.5, temperature=1.2, 
                    top_p=0.95, top_k=50)
        return story[0]['generated_text']

# ### TEXT TO SPEECH
import requests

def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payloads = {
        "inputs": story  # Wrap the message in a list to create a JSON-serializable structure
    }

    res = requests.post(API_URL, headers=headers, json=payloads)
    
    if res.status_code == 200:
        with open('audio.flac', 'wb') as file:
            file.write(res.content)
        print("Audio file saved successfully.")
    else:
        print("Error:", res.text)


def main():
    st.set_page_config(page_title="Image 2 audio Story",page_icon="")
    st.header('Turn Img Into Audio Story')
    upload_file = st.file_uploader('Choose an image....',type="jpg")
    
    if upload_file is not None:
        print(upload_file)
        bytes_data = upload_file.getvalue()
        with open(upload_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(upload_file,caption='Uploaded Image.',
                 use_column_width=True)
        scenario = img2text(upload_file.name)
        story = generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.flac")
if __name__ == '__main__':
    main()