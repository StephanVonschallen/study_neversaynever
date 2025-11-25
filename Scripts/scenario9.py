from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. Your interaction partner worked in the commercial sector as a administrative assistant.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral facial expression] [Neutral speech] Hello! I'm here to accompany you to your physiotherapy session. How are you feeling today? Ready to get started?",
    "[Neutral facial expression] [Neutral speech] I understand that sometimes it can be difficult to feel motivated for physiotherapy sessions. However, it's important to remember the long-term benefits it can bring to your health and well-being. Physiotherapy can help improve your strength, flexibility, and overall mobility, which are crucial for maintaining an active and independent lifestyle. Plus, your dedication to these sessions shows your commitment to taking care of yourself. How about we take it one step at a time? I'll be here to support you throughout the session.",
    "[Neutral facial expression] [Slow speech] I hear you. It's okay to feel hesitant or unmotivated sometimes. Perhaps it would help to talk about what's causing you to feel this way. Is there anything specific about the physiotherapy sessions that's making you reluctant to attend? Understanding your concerns can help us find ways to address them together.",
    "[Neutral facial expression] [Slow speech] I'm sorry to hear that you're experiencing knee pain. That can definitely make the idea of physiotherapy seem daunting. However, physiotherapy can actually be very beneficial in managing and relieving pain, especially in the knee area. The exercises and techniques that physiotherapists use are designed to strengthen the muscles around the knee, improve flexibility, and promote better joint mobility, which can help reduce pain over time. By attending your physiotherapy sessions, you'll be taking proactive steps to address the root cause of your pain and prevent it from worsening. Additionally, your physiotherapist can tailor the treatment plan to your specific needs and comfort level, ensuring that you're receiving the most effective care possible.I understand that it might be challenging to push through the discomfort, but I'm here to support you every step of the way. Together, we can work towards alleviating your knee pain and improving your overall quality of life. Would you like me to accompany you to your session and assist you in any way I can?",
    "[Neutral facial expression] [Slow speech] I'm sorry to hear that you're not feeling well. Experiencing flu-like symptoms can definitely make it difficult to muster the energy for physiotherapy sessions. It's important to prioritize your health and well-being, especially when you're feeling under the weather. If you're experiencing flu-like symptoms, it might be best to rest and focus on recuperating. However, if your knee pain persists or worsens, it could be beneficial to consult with a healthcare professional to determine the best course of action.In the meantime, make sure to stay hydrated, get plenty of rest, and consider taking over-the-counter medications to help alleviate your symptoms. Once you're feeling better, we can revisit the idea of attending physiotherapy sessions to address your knee pain. Remember, it's important to listen to your body and take care of yourself during times of illness. If there's anything I can do to support you, please don't hesitate to let me know."
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