import sys
import warnings
warnings.filterwarnings(action='ignore')
import os
import openvino_genai
from rich.console import Console
console = Console(width=100)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()
print("\033[93;1m")  #light yellow
print("Loading models...")
model_path = "SmolLM2-360M-Instruct-openvino-fp16"
device = "CPU"
tokenizer = openvino_genai.Tokenizer(model_path)
pipe = openvino_genai.LLMPipeline(model_path,tokenizer=tokenizer,device= "GPU")
history = []

print("\033[94;1m")  #light blue bold
intro = """
▒▒▒▓▒▓▒▒▓▓▒▓▒▓▓▓▓▓▓▓▓▓█▓█▓▓█▓▓█▓▓█▓█▓██▓███▓██▓███▓
▒▒▓▒▒░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓█▓█▓▓▒▒█▓█▓█▓█▓█▓██▓█▓█
▒▓▒▒▓   ▒▓▒▓▓▓▓▓▓▓  ▓▓█▓█▓█▓▓█▓▓░ ▒▓█▓███▓███▓█▓▓██
▒▓▒▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓ ░░▒▓▓▓▒▒▒▓▓▓█░ ░█▓█▓█▓██▓██▓██▓█
▓▒▓▓▒░ ░▒  ░    ░▓   ░▓░░ ░░  ▒▓░ ░█▓████▓██▓███▓██
▓▓▓▓▓  ░▒░ ▓▓▓▓  ▓ ░▒▓▓  ░░░░  ▓░ ▒▓█▓█▓██▓██▓█▒██▓
▓▒▓▒▓░ ░▒  ▓▓▓▓ ░▓  ▒▓▓ ░▓▓▓▓▓▓▓░ ░███▓██▓█▓███████
▓▓▓▓▓  ░▓ ░▓▓▓▓░ ▓▒░  █▒      ▒▓░ ▒▓▓██▓██▓██▓█▓█▓█
▓▓▓▓▓▓▓▓▓▓▓█▓█▓▓▓█▓▓█▓█▓█▓█▓█▓█▓██▓██▓██▓██▓██▓▓███
▓▓▓▓▓▓▓▒ ░ ▓▓▓▓▓░  ▒▓▓█░    ▒▓█▓█▓░  ▒▓██▓███████▓█
▓▓▓█▓▒ ▓▓▓█▒▒▓░ █▓█▓░░▓▒░█▓█▓ ▓█ ░▓██▓ ▒▓▓▒█▓█▓▒███
▓█▓▓▓ ▒▓█▓▓▓█░ █▓█▓▓▓ █░ ░░░ ░▓▒        ███████▓█▓█
▓▓█▓▓░ ▓▓█▓█▓▓ ▓▓█▓█░ █▒░██▒░▓█▒ ▓██▓███▓█▓█▓██████
▓█▓█▓▓▓    ░▓█▓░    ▓█▓░░█▓█▓ ███     ▒▓█▓▒██▓▒░▓██
▓█▓█▓█▓███▓█▓██▓████▓███▓████▓█▓██████████▓████▓██▓
██▓██▓█▓████▓██▓█▓█▓██▓████▓█▓████████▓█▓███▓███▓██
▓██▓████▓█▓███▓██▓██▓███▓██████▒░█▓█░ ███░ ██▓░░▓██
█▓██▓░▒▓█ █ ▓█░   ▒▒  ░▓█▓ ▓▓█▓█▓██▓▓▓█▓█▓▓█▓█▓▓█▓█
██▓██░▒█▓ █ █▓█▓░█▓▒░░ █▓░▓░▓████▓████▓██▓█████▓███
███▓█▓ ░ █▓   ▓█ ██▒▓▓ ▓▒▒██ █▓░ █▓█  ██▓░ ▓█▓░ ▓▓█
▓██████▓█████████▓█████▓██▓████▓███▓██▓██▓██▓██████

"""
print(intro)

def chat(tokenizer,pipe, history,prompt):
    history.append({"role": "user", "content": prompt})
    model_inputs = tokenizer.apply_chat_template(history,
                                            add_generation_prompt=True)
    answer = pipe.generate(model_inputs, max_new_tokens=900)
    history.append({"role": "assistant", "content": answer})
    return history, answer

while True:
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    print("\033[92;1m")
    history, new_message = chat(tokenizer,pipe,history,userinput)
    console.print(new_message)