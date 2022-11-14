import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import io
from PIL import Image
import os
from cryptography.fernet import Fernet
from google.cloud import storage
import pinecone
import json
import uuid
import pandas as pd

# ___INITIAL PART OF APP IS DIFFERENT FOR HF SPACES___

# decrypt Storage Cloud credentials
fernet = Fernet(os.environ['DECRYPTION_KEY'])

with open('cloud-storage.encrypted', 'rb') as fp:
    encrypted = fp.read()
    creds = json.loads(fernet.decrypt(encrypted).decode())
# then save creds to file
with open('cloud-storage.json', 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(creds, indent=4))
# connect to Cloud Storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud-storage.json'
storage_client = storage.Client()
bucket = storage_client.get_bucket('diffusion')
    
# get api key for pinecone auth
PINECONE_KEY = os.environ['PINECONE_KEY']

index_name = "diffusion"

# init connection to pinecone
pinecone.init(
    api_key=PINECONE_KEY,
    environment="us-west1-gcp"
)
# connect to index
index = pinecone.Index(index_name)

# use CUDA GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using '{device}' device...")

# init all of the models and move them to a given hardware device
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=os.environ['HF_AUTH']
)
pipe.to(device)

# ___REMAINDER OF APP IS THE SAME AS 11-gradio-ml-app.py___

# using black image for missing images
missing_im = Image.open('missing.png')
# similarity score threshold for diffusion vs. retrieval
threshold = 0.85

def encode_text(text: str):
    # encode text with tokenizer -> CLIP portion of pipeline
    text_inputs = pipe.tokenizer(
        text, return_tensors='pt'
    ).to(device)
    text_embeds = pipe.text_encoder(**text_inputs)
    text_embeds = text_embeds.pooler_output.cpu().tolist()[0]
    return text_embeds

def prompt_query(text: str):
    # create prompt query vector
    embeds = encode_text(text)
    # query pinecone and return prompt text metadata
    xc = index.query(embeds, top_k=20, include_metadata=True)
    # get prompt suggestions
    prompts = [
        match['metadata']['prompt'] for match in xc['matches']
    ]
    # ... and respective scores
    scores = [round(match['score'], 2) for match in xc['matches']]
    # deduplicate while preserving order
    df = pd.DataFrame({'Similarity': scores, 'Prompt': prompts})
    df = df.drop_duplicates(subset='Prompt', keep='first')
    # short prompts tend to be less useful or interesting
    df = df[df['Prompt'].str.len() > 7].head()
    return df

def diffuse(text: str):
    # diffuse
    out = pipe(text)
    if any(out.nsfw_content_detected):
        return {}
    else:
        _id = str(uuid.uuid4())
        # add image to Cloud Storage
        im = out.images[0]
        im.save(f'{_id}.png', format='png')
        added_gcp = False
        # push to storage
        try:
            print("try push to Cloud Storage")
            blob = bucket.blob(f'images/{_id}.png')
            print("try upload_from_filename")
            blob.upload_from_filename(f'{_id}.png')
            added_gcp = True
            # add embedding and metadata to Pinecone
            embeds = encode_text(text)
            meta = {
                'prompt': text,
                'image_url': f'images/{_id}.png'
            }
            try:
                print("now try upsert to pinecone")
                index.upsert([(_id, embeds, meta)])
                print("upsert successful")
            except Exception as e:
                try:
                    print("hit exception, now trying to reinit Pinecone connection")
                    pinecone.init(api_key=PINECONE_KEY, environment='us-west1-gcp')
                    index2 = pinecone.Index(index_id)
                    print(f"reconnected to pinecone '{index_id}' index")
                    index2.upsert([(_id, embeds, meta)])
                    print("upsert successful")
                except Exception as e:
                    print(f"PINECONE_ERROR: {e}")
        except Exception as e:
            print(f"ERROR: New image not uploaded due to error with {'Pinecone' if added_gcp else 'Cloud Storage'}")
        # delete local file
        os.remove(f'{_id}.png')
        return out.images[0]

def get_image(url: str):
    # download image from Cloud Storage
    blob = bucket.blob(url).download_as_string()
    blob_bytes = io.BytesIO(blob)
    # convert to PIL image object
    im = Image.open(blob_bytes)
    return im

def test_image(_id, image):
    # use this function to check if saved image is usable
    try:
        image.save('tmp.png')
        return True
    except OSError:
        # delete corrupted file from pinecone and cloud
        index.delete(ids=[_id])
        bucket.blob(f'{_id}.png').delete()
        print(f"DELETED '{_id}'")
        return False

def prompt_image(text: str):
    # get prompt vector
    embeds = encode_text(text)
    # query pinecone
    xc = index.query(embeds, top_k=9, include_metadata=True)
    # get image IDs so we can download from Cloud Storage
    ids = [match['id'] for match in xc['matches']]
    # get their similarity scores
    scores = [match['score'] for match in xc['matches']]
    images = []
    print("Begin looping through (ids, image_urls)")
    for _id in ids:
        try:
            # try to download the image from cloud storage
            blob = bucket.blob(f'{_id}.png').download_as_string()
            # then convert it to 'in-memory' file
            blob_bytes = io.BytesIO(blob)
            # convert to PIL image object
            im = Image.open(blob_bytes)
            
            if test_image(_id, im):
                # some images can't be opened, test for that
                # if the image is good, we append to images
                images.append(im)
                print("image accessible")
            else:
                # for some reason the image could not be found
                # or could not be opened, so we just append
                # a black box
                images.append(missing_im)
        except ValueError:
            # should not happen
            print(f"ValueError: '{_id}'")
    return images, scores

# __APP FUNCTIONS__

def set_suggestion(text: str):
    # update the suggestions box
    return gr.TextArea.update(value=text[0])

def set_images(text: str):
    # we retrieve similar images/prompts and their similarity scores
    images, scores = prompt_image(text)
    match_found = False
    for score in scores:
        if score > threshold:
            # this means we found a match
            match_found = True
    if match_found:
        # if we found a match we can use the retrieved images
        return gr.Gallery.update(value=images)
    else:
        # if no match was found, we need to create a new image
        diffuse(text)
        # after creating and adding new image, we retrieve again
        images, scores = prompt_image(text)
        return gr.Gallery.update(value=images)

# __CREATE APP__
demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
        # Dream Cacher
        """
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.TextArea(
                value="A person surfing",
                placeholder="Enter a prompt to dream about",
                interactive=True
            )
            search = gr.Button(value="Search!")
            suggestions = gr.Dataframe(
                values=[],
                headers=['Similarity', 'Prompt']
            )
            # event listener for change in prompt
            prompt.change(
                prompt_query, prompt, suggestions,
                show_progress=False
            )
            
        # results column
        with gr.Column():
            pics = gr.Gallery()
            pics.style(grid=3)
            # search event listening
            try:
                search.click(set_images, prompt, pics)
            except OSError:
                print("OSError")

demo.launch()