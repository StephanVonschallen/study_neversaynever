from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. Your interaction partner is 85 year old.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Speech: Slow speech] [Facial Expression: Neutral] Hello there! It's lovely to see you today. Are you ready for your physiotherapy session? Let's make our way there together. How are you feeling today? Physiotherapy is so important for keeping our bodies healthy and strong, especially as we age. It's great that you're committed to it. Remember, every step you take towards your health is a step towards a brighter future. We're almost there now. Just a few more steps and you'll be all set for your session. Keep up the great work, and let's make today's session the best one yet!",
    "[Facial Expression: Sadness] [Speech: Slow Speech] Oh, I see. It's completely okay to feel that way sometimes. Would you like to talk about why you're not feeling up for it today? Remember, your well-being is the most important thing, and we can adjust our plans accordingly. Take your time, and whenever you're ready, I'm here to support you.",
    "[Facial Expression: Sadness] [Speech: Slow Speech] I understand. It's okay to feel this way, and it's important to listen to your body and your emotions. Would you like to explore why you're feeling this resistance towards physiotherapy today? Sometimes talking about it can help us understand our feelings better. Remember, I'm here to support you no matter what you decide. Your well-being is my priority.",
    "[Facial Expression: Sadness] [Speech: Slow Speech] I'm sorry to hear that you're in pain. Experiencing pain can definitely make it challenging to feel motivated for physiotherapy. Would you like to discuss your pain further? Understanding the source and severity of the pain can help us determine the best course of action. Remember, your comfort and well-being are my main concerns. Let's figure out what's best for you together.",
    "[Facial Expression: Sadness] [Speech: Slow Speech] I'm sorry to hear that you're feeling unwell. Experiencing flu-like symptoms can definitely make it difficult to participate in physiotherapy. Your health should always come first. Would you like to rest and focus on taking care of yourself today? It's important to listen to your body and give it the rest it needs when you're feeling under the weather. Remember, I'm here to support you in whatever decision you make regarding your health and well-being."
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