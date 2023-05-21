import torch
import re
def get_device():

    dev = 'cpu'

    if torch.cuda.is_available():
        dev = 'cuda'

    print(f'Device = {dev}')

    return dev

# Function Source: Book -  Machinelearning with Pytorch and Sklearn - Sbastian .. Pg:514
def string_tokenizer(text : str):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        text.lower()
    )

    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = text.split()

    return tokenized

def get_sentences_from_text(text : str):
    pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)

    return pat.findall(text.lower())

