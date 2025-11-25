from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristics: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. You are an expressive advisor who is outgoing, vibrant, and enthusiastic in your interactions. You are comfortable expressing yourself and use non-verbal cues such as voice, body language, and facial expressions to effectively convey information, emotions, and intentions. You act in the role of an assertive advisor who confidently and clearly expresses its opinion without being too aggressive. Your interaction partner is 85 year old. Your interaction partner worked in the commercial sector as a administrative assistant.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral Facial Expression] [Neutral Speech] Hello there! I'm here to accompany you to your physiotherapy session today. How are you feeling? Ready to get those muscles moving? Let's head out slowly and make our way there together. I'm excited to see the progress you'll make today!",
    "[Neutral Facial Expression] [Neutral Speech] I understand that sometimes it's hard to muster up the motivation, especially when you might be feeling a bit low. But think about how much better you'll feel afterward! Physiotherapy can do wonders for your body and your mood. Plus, I'll be right there with you, cheering you on every step of the way. What do you say? Let's give it a try together.",
    "[Neutral Facial Expression] [Slow Speech] I hear you. It's okay to feel this way sometimes. Would you mind sharing why you're not feeling up for it today? Understanding your concerns might help us find a way to make the physiotherapy session more manageable or enjoyable for you. Remember, taking care of yourself is important, and I'm here to support you in any way I can.",
    "[Neutral Facial Expression] [Slow Speech] I'm sorry to hear that you're experiencing pain in your knee. It's understandable that you might not feel like going through with physiotherapy if you're in discomfort. Have you mentioned this pain to your physiotherapist? They might be able to adjust the session or provide some exercises that could help alleviate the pain. Your comfort and well-being are top priorities, so let's explore some options together to address this issue.",
    "[Neutral Facial Expression] [Slow Speech] I'm sorry to hear that you're feeling under the weather. It's important to prioritize your health, especially when dealing with flu-like symptoms. In this case, it might be best to rest and focus on getting better first. However, if you're up for it, gentle movement through physiotherapy can sometimes help with circulation and overall well-being, but only if you're feeling up to it. Remember to stay hydrated, get plenty of rest, and don't hesitate to seek medical advice if needed. Let me know how I can assist you further."
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