from typing import List, Dict, Union


def split_messages(chat: str) -> Dict[str, Union[List[Dict[str, str]], str]]:
    """
    {
        "conversations": [
            {
                "from": "human",
                "value": "用户指令"
            },
            {
                "from": "gpt",
                "value": "模型回答"
            }
        ],
        "system": "系统提示词（选填）"
    }
    """
    system = ""
    split_marks = ["\n\nHuman: ", "\n\nAssistant: "]
    conversations = []
    while split_marks[0] in chat or split_marks[1] in chat:
        if chat.startswith(split_marks[0]):
            index = chat.find(split_marks[1])
            if index == -1:
                value, chat = chat, ''
            else:
                value, chat = chat[:index], chat[index:]
            conversation = {
                "from": "human",
                "value": value[len(split_marks[0]):]
            }
        elif chat.startswith(split_marks[1]):
            index = chat.find(split_marks[0])
            if index == -1:
                value, chat = chat, ''
            else:
                value, chat = chat[:index], chat[index:]
            conversation = {
                "from": "gpt",
                "value": value[len(split_marks[1]):]
            }
        else:
            raise AssertionError(f"Chat Error: {chat}")
        conversations.append(conversation)

    for i, conversation in enumerate(conversations):
        assert (i % 2 == 0 and conversation["from"] == "human") or (i % 2 == 1 and conversation["from"] == "gpt")

    return {
        "conversations": conversations,
        "system": system
    }


def openchat_format_chat(conversations: List) -> List[Dict[str, str]]:
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you today?"}
    ]
    """
    machine_name =  ['gpt', 'assistant']
    messages = []
    for conversation in conversations:
        message = {
            "role": 'assistant' if conversation["from"].lower() in machine_name else 'user',
            "content": conversation["value"],
        }
        messages.append(message)
    return messages


def build_openchat_prompt(messages: List, eos_token='<|end_of_turn|>') -> str:
    """
    GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:
    """
    prompt = ''
    for i, message in enumerate(messages):
        if i % 2 == 0:
            prompt += f"GPT4 Correct User: {message['content']}{eos_token}GPT4 Correct Assistant:"
        elif message['content']:
            prompt += f" {message['content']}{eos_token}"
    return prompt


def build_zephyr_prompt(messages: List, eos_token='</s>', system="") -> str:
    """
    <|system|>
    You are a friendly chatbot who always responds in the style of a pirate.</s>
    <|user|>
    How many helicopters can a human eat in one sitting?</s>
    <|assistant|>
    Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!
    """
    prompt = f"<|system|>\n{system}{eos_token}\n"
    for i, message in enumerate(messages):
        if i % 2 == 0:
            prompt += f"<|user|>\n{message['content']}{eos_token}\n<|assistant|>\n"
        elif message['content']:
            prompt += f"{message['content']}{eos_token}\n"
    return prompt


def build_mistral_prompt(messages: List, eos_token='</s>', system="") -> str:
    """
    <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
    """
    prompt = f"{system}"
    for i, message in enumerate(messages):
        if i % 2 == 0:
            prompt += f" [INST] {message['content'].strip(' ')} [/INST]"
        elif message['content']:
            prompt += f" {message['content'].strip(' ')}{eos_token}"
    return prompt


def convert_format_from_hh_to_openchat(prompt: str, eos_token='<|end_of_turn|>') -> str:
    return build_openchat_prompt(openchat_format_chat(split_messages(prompt)['conversations']), eos_token)


def build_llama_input(sample: Dict) -> str:
    conversations = sample["conversations"]
    messages = [(conversation['from'], conversation['value']) for conversation in conversations]

    input_str = ""
    for idx, (role, message) in enumerate(messages):
        if idx % 2 == 0:
            input_str += f"<s>[INST] {message} [/INST]"
        else:
            input_str += f" {message} </s>"
    return input_str
