from model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import time

base_model_path = "/home/ubuntu/model_input/Llama-3-8B/model_int4.g32.pth"
eagle_model_path = "/home/ubuntu/model_input/eagle-llama2/model_int4.g32.pth"

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=eagle_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda"
)

model.eval()

model.draft_one=torch.compile(model.draft_one, mode="reduce-overhead", fullgraph=True,dynamic=False)
model.base_forward=torch.compile(model.base_forward, mode="reduce-overhead", fullgraph=True,dynamic=False)
model.base_forward_one=torch.compile(model.base_forward_one, mode="reduce-overhead", fullgraph=True)

your_message="write an article on India"

conv = get_conversation_template("llama-2-chat")  
sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
conv.system_message = sys_p
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt() + " "

for _ in range(10):
    st = time.time()
    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model.eagenerate(input_ids,temperature=0,max_new_tokens=128)
    output=model.tokenizer.decode(output_ids[0])
    et = time.time()

    out_tokens = len(model.tokenizer.encode(output))-len(model.tokenizer.encode(prompt))
    print(et-st, out_tokens, out_tokens/(et-st))