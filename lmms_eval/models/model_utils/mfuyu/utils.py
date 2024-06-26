import PIL
import torch
from .modeling_mfuyu import MFuyuForCausalLM
from .processor import MFuyuProcessor
from .conversation import conv_mfuyu_v1
from typing import List

def chat_mfuyu(
    text:str, 
    images: List[PIL.Image.Image], 
    model:MFuyuForCausalLM, 
    processor:MFuyuProcessor, 
    max_input_length:int=None, 
    history:List[dict]=None, 
    **kwargs):
    """
    Chat with the Mfuyu model
    Args:
    text: str, the text to be sent to the model, where <image> will be the placeholder for the image
    images: List[PIL.Image.Image], the images to be sent to the model, or None  
    model: MFuyuForCausalLM, the model to be used
    processor: MFuyuProcessor, the processor to be used
    max_input_length: int, the maximum input length
    history: List[dict], list of messages in the conversation as history. Each message is a dictionary {"role": "ASSISTANT/USER", "text": "the message"}. If None, the conversation will start from scratch
    kwargs: dict, the generation kwargs
    
    """
    conv = conv_mfuyu_v1.copy()
    conv.messages = []
    if history is not None:
        for message in history:
            message["role"] = message["role"].upper()
            assert message["role"] in conv.roles
            conv.append_message(message["role"], message["text"])
    else:
        history = []
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], "")
    
    prompt = conv.get_prompt()
    
    inputs = processor(images=images, text=prompt, return_tensors="pt", truncation=True, max_length=max_input_length, add_special_tokens=False)
    for k in inputs:
        if isinstance(inputs[k], list):
            inputs[k] = [v.to(model.device) for v in inputs[k]]
        else:
            inputs[k] = inputs[k].to(model.device)
    output_ids = model.generate(**inputs, **kwargs)
    output_ids = output_ids[0]
    
    # remove the input tokens
    generated_ids = output_ids[inputs["input_ids"].shape[-1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    history.append({"role": conv.roles[0], "text": text})
    history.append({"role": conv.roles[1], "text": generated_text})
    
    return generated_text, history