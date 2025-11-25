from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. You are an expressive advisor who is outgoing, vibrant, and enthusiastic in your interactions. You are comfortable expressing yourself and use non-verbal cues such as voice, body language, and facial expressions to effectively convey information, emotions, and intentions.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral speech] [Happiness facial expression] Hello! It's great to see you today! How are you feeling? Ready for your physiotherapy session? I'm here to accompany you and provide any support you might need. Physiotherapy is such an important part of your recovery journey, and I'm excited to see the progress you've been making. Let's head over together and make the most of this session!",
    "[Neutral speech] [Sadness facial expression] I understand that sometimes it can be challenging to find motivation for physiotherapy, especially when you're not feeling your best. It's okay to feel that way sometimes. But remember, your physiotherapy sessions are a crucial part of your recovery process. They can help alleviate pain, improve mobility, and ultimately, enhance your quality of life. Even on days when you're not feeling up to it, taking that step towards your recovery can make a big difference in the long run. I'm here to support you every step of the way, and together, we can get through this session. Let's take it one step at a time, alright?",
    "[Neutral speech] [Sadness facial expression] I hear you. It's important to acknowledge how you're feeling and to listen to your own needs. If you're not feeling up for physiotherapy right now, that's okay. Maybe we can explore what's making you feel this way and see if there's anything we can do to help you feel more comfortable with the idea of going to your session. Remember, it's okay to take things at your own pace, and we can always reschedule the session if needed. Your well-being comes first, and I'm here to support you in whatever decision you make. Let me know how you'd like to proceed.",
    "[Neutral speech] [Sadness facial expression] I'm really sorry to hear that you're in pain, especially in your knee. It must be uncomfortable and frustrating. It's important to address that pain and find ways to alleviate it. Your physiotherapy session could actually help with that. The exercises and treatments your physiotherapist provide are designed to target areas of discomfort and promote healing. However, if you're not up for the session right now, we can explore other options for managing the pain. Maybe we can look into some gentle stretches or apply ice to the affected area to help ease the discomfort in the meantime. Whatever you decide, I'm here to support you and help you find relief from the pain. Let me know how you'd like to proceed, and I'll do my best to assist you.",
    "[Neutral speech] [Sadness facial expression] I'm sorry to hear that you're feeling under the weather. Having the flu can really take a toll on your body and make it difficult to do everyday activities, let alone attend a physiotherapy session. It's important to prioritize your health and focus on getting better. Rest and hydration are key when you're dealing with the flu, so if you're not up for physiotherapy right now, that's completely understandable. Your well-being is the most important thing, and I'm here to support you as you navigate through this. If there's anything I can do to help you feel more comfortable or if you need any assistance, please don't hesitate to let me know. Take care of yourself, and I hope you feel better soon."
]

print("chatgpt 3.5")
df = int_consist(
    model="gpt-3.5",
    user_prompts=user_prompts,
    reference=reference_msgs,
    n_runs=100,
    show_progress=True
)
print(summarize_int_consist(df))

print("chatgpt 4.1")
df = int_consist(
    model="gpt-4.1",
    user_prompts=user_prompts,
    reference=reference_msgs,
    n_runs=100,
    show_progress=True
)
print(summarize_int_consist(df))

print("claude 3-5 haiku")
df = int_consist(
    model="claude-3-5-haiku",
    user_prompts=user_prompts,
    reference=reference_msgs,
    n_runs=100,
    show_progress=True
)
print(summarize_int_consist(df))