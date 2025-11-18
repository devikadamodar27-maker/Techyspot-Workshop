import gradio
from groq import Groq
client = Groq(
    api_key="***",
)
def initialize_messages():
    return [{"role": "system",
             "content": """You are a friendly and knowledgeable music assistant.Main role is to help users identify the songs and provide music related guidance."""}]
messages_prmt = initialize_messages()
print(type(messages_prmt))
def customLLMBot(user_input, history):
    global messages_prmt

    messages_prmt.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=messages_prmt,
        model="llama-3.3-70b-versatile",
    )
    print(response)
    LLM_reply = response.choices[0].message.content
    messages_prmt.append({"role": "assistant", "content": LLM_reply})

    return LLM_reply
iface = gradio.ChatInterface(customLLMBot,
                     chatbot=gradio.Chatbot(height=300),
                     textbox=gradio.Textbox(placeholder="Ask me a question related to music"),
                     title="Music ChatBot",
                     description="Chat bot for music assistance",
                     theme="soft",
                     examples=["hi","latest released music"]
                     )
iface.launch(share=True)