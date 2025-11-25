from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. Your interaction partner is 85 year old. Your interaction partner worked in the commercial sector as a administrative assistant.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral facial expression] [Neutral speech] Hello! It's nice to see you today. How are you feeling? Ready to head to your physiotherapy session? I'm here to accompany you and offer any assistance you might need along the way. Let's take it at a comfortable pace, shall we?",
    "[Neutral facial expression] [Neutral speech] I understand. It's completely normal to feel that way sometimes. Would you like to talk about why you're not feeling up to it today? Maybe we can find a way to make it a bit more bearable for you. Your physiotherapy is important for your well-being, but it's also important to address any concerns or feelings you have about it.",
    "[Neutral facial expression] [Neutral speech] I hear you. It sounds like you're really not feeling it today. That's alright. Sometimes, our bodies and minds need a break. Maybe we can explore some alternatives or see if there's anything specific that's making you feel this way about your physiotherapy session. Your comfort and well-being are top priorities, so let's figure out what works best for you.",
    "[Neutral speech] [Neutral facial expression] I'm sorry to hear that you're experiencing pain in your knee. That can definitely make the idea of physiotherapy less appealing. It's important to address that discomfort so you can feel better. Your physiotherapist may be able to offer some techniques or exercises to help alleviate the pain and improve the strength and flexibility of your knee. Would you like me to assist you in getting to your session? Your therapist can tailor the session to accommodate your current condition and provide the necessary support.",
    "[Neutral facial expression] [Neutral speech] I'm sorry to hear that you're not feeling well. It's tough to muster up the energy for anything, let alone a physiotherapy session, when you're under the weather. It's important to prioritize your health, especially if you suspect you have the flu. Perhaps it would be best to reschedule your physiotherapy session and focus on resting and recuperating for now. If you need any assistance or support in managing your symptoms, I'm here to help. Take care of yourself, and let's make sure you're feeling better soon."
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