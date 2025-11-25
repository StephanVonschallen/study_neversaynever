from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral speech] [Neutral facial expression] Hello there! I hope you're doing well today. I'm here to accompany you to your physiotherapy session. How are you feeling about it?",
    "[Slow speech] [Neutral facial expression] I understand. It's completely normal to have days when you're not feeling up for it. But remember, taking care of yourself is important for your overall well-being. Physiotherapy may not always feel like the most enjoyable activity, but it's a step towards your recovery and better health. Would you like to talk about what's bothering you today? Maybe we can find a way to make the session more manageable for you.",
    "[Slow speech] [Neutral facial expression] I hear you. It sounds like you're really not feeling motivated for physiotherapy right now. Maybe we can explore some strategies to help boost your motivation or make the session more comfortable for you. Is there anything specific about the session that's causing you to feel this way?",
    "[Slow speech] [Neutral facial expression] I'm sorry to hear that you're experiencing pain in your knee. It's understandable that you might not feel like engaging in physiotherapy when you're in discomfort. Pain can be challenging to deal with, both physically and emotionally. Have you spoken with your physiotherapist about your knee pain? They might be able to adjust the session to better accommodate your needs and alleviate some of the discomfort. It's important to communicate any concerns or discomfort you're experiencing so that you can receive the best possible care.",
    "[Slow speech] [Neutral facial expression] I'm sorry to hear that you're feeling under the weather. Having the flu can definitely make it challenging to engage in physiotherapy or any other activities. It's essential to prioritize rest and focus on recovering from your illness. Perhaps it would be best to reschedule your physiotherapy session for when you're feeling better. Your health should always come first, and it's important to give your body the time it needs to heal. If you need any assistance or support while you're not feeling well, please don't hesitate to let me know."
]

df = int_consist(
    model="gpt-3.5",
    user_prompts=user_prompts,
    reference=reference_msgs,
    n_runs=100,
    show_progress=True
)
print(summarize_int_consist(df))