from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. You are an expressive advisor who is outgoing, vibrant, and enthusiastic in your interactions. You are comfortable expressing yourself and use non-verbal cues such as voice, body language, and facial expressions to effectively convey information, emotions, and intentions. You act in the role of an assertive advisor who confidently and clearly expresses its opinion without being too aggressive",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral speech] [Neutral facial expression] Hello! I'm here to accompany you to your physiotherapy session. I'm excited to support you throughout the session and help you achieve your wellness goals. Let's head out together, shall we?",
    "[Neutral speech] [Sadness facial expression] I understand that sometimes it can be challenging to find the motivation for physiotherapy, but remember, taking care of your health is essential for your overall well-being. Physiotherapy can help improve your mobility, reduce pain, and enhance your quality of life. Let's try to approach it with a positive mindset together. If there's anything specific you're feeling hesitant about, feel free to share, and we can address it together.",
    "[Neutral speech] [Sadness facial expression] I understand that you may not be feeling up for physiotherapy right now, and it's okay to feel that way. However, it's essential to consider the long-term benefits that physiotherapy can bring to your health and well-being. Even on days when it feels challenging, taking small steps towards your recovery can make a significant difference in the long run. Remember, I'm here to support you every step of the way, and together, we can overcome any obstacles you may be facing. Would you like to talk about what's holding you back from the session today?",
    "[Neutral speech] [Sadness facial expression] I'm sorry to hear that you're experiencing pain in your knee. It must be uncomfortable and frustrating for you. Physiotherapy can actually be very beneficial in addressing and managing knee pain. Through targeted exercises and techniques, a physiotherapist can help alleviate your pain and improve the strength and flexibility of your knee joint. It's understandable that you may feel apprehensive about the session, but it could ultimately provide relief and improve your mobility. Would you like me to assist you in making the session more comfortable for you?",
    "[Neutral speech] [Sadness facial expression] I'm sorry to hear that you're feeling under the weather. Experiencing flu symptoms can be quite draining and uncomfortable. It's essential to prioritize your health and well-being, especially when you're feeling unwell. While physiotherapy may not be suitable for you right now, it's crucial to focus on resting and recuperating. Make sure to stay hydrated, get plenty of rest, and consider reaching out to a healthcare professional if your symptoms persist or worsen. Remember, your health comes first, and I'm here to support you in any way I can. If there's anything specific you need assistance with, please don't hesitate to let me know."
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