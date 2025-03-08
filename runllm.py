import PIL.Image
import os
import torch
from vllm import LLM,SamplingParams
from BRBS.models.Vit import Reconstruct

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

class Detect:

    def __init__(self, model_path, temperature=0.8, top_p=0.95):
        if not isinstance(temperature, (int, float)) or not isinstance(top_p, (int, float)):
            raise ValueError('"temperature"  "top_p" int or float')

        if temperature < 0:
            raise ValueError('"temperature" [0, +∞]')

        if top_p <= 0 or top_p > 1:
            raise ValueError(' "top_p" (0, 1]')

        self.template = 'hi'
        self.llm = LLM(model=model_path, dtype=torch.bfloat16, gpu_memory_utilization=0.9, task="reward")
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    def generate_messages_with_path(self, image_path):
        base64_image = PIL.Image.open(image_path)
        prompt = "USER: <image>\n Describe this picture\nASSISTANT:"
        messages = {"prompt": prompt, "multi_modal_data": {"image": base64_image}, }
        return messages

    def generate_messages(self, image):

        prompt = "USER: <image>\n Describe the disease information in the picture\nASSISTANT:"
        messages = {"prompt": prompt, "multi_modal_data": {"image": image}, }
        return messages

    def detect(self, messages):

        # outputs = self.llm.generate(messages)
        encode_outputs = self.llm.encode(messages)
        # return [output.outputs[0].text for output in outputs]
        temp = encode_outputs[0].outputs.data
        return temp


if __name__ == '__main__':
    model_path = "./saves/llava-1.5-7b-lora"
    myLLM = Detect(model_path)
    image_path = ""  
    messages = myLLM.generate_messages_with_path(image_path)
    hidden_result = myLLM.detect(messages)
    x = hidden_result.unsqueeze(0)

    print(hidden_result)