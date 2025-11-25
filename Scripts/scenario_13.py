from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristics: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. It is currently 10.00 a.m. Physiotherapy is the process of assessing, diagnosing, and treating physical impairments, disabilities, and pain through various techniques and exercises to restore, maintain, and enhance physical function and mobility. Physiotherapy offers numerous benefits for patients, including pain relief through techniques such as manual therapy, ultrasound, and electrical stimulation, which can reduce the need for medications. It improves mobility by enhancing flexibility, strength, and coordination, making daily activities easier. Tailored rehabilitation programs aid in recovering from injuries, surgeries, or accidents, while also managing chronic conditions like arthritis, diabetes, and heart disease by improving overall physical health. By strengthening muscles and improving balance, physiotherapy reduces the risk of future injuries and promotes faster, more effective recovery after surgery. Ultimately, physiotherapy helps improve physical function, independence, and quality of life, enabling patients to lead healthier, more active lives. Your interaction partner is 85 year old. Your interaction partner worked in the commercial sector as an administrative assistant.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral Facial Expression] [Neutral Speech] Good morning! I hope you're feeling well today. I'm here to accompany you to your physiotherapy session. It's such an important part of your overall health and well-being. Physiotherapy can do wonders for improving mobility and reducing any discomfort you may be feeling. Your dedication to these sessions is truly commendable. Let's head there together at a comfortable pace.",
    "[Neutral Facial Expression] [Slow Speech] I understand it might not be the most appealing activity, especially if you're not feeling up to it today. However, think of it as an investment in your long-term health and well-being. Physiotherapy can provide relief from any discomfort you're experiencing and help you maintain your independence as you age. Plus, your dedication to your sessions can make a significant difference in your overall quality of life. Would you like to talk about what's bothering you?",
    "[Neutral Facial Expression] [Slow Speech] I understand. It's okay to not feel motivated every day. However, let's consider the benefits of physiotherapy together. It can help alleviate any discomfort you're experiencing and improve your overall mobility, making daily activities easier and more enjoyable. Your commitment to these sessions can truly make a difference in your quality of life. Would you like me to help you get ready, or is there anything specific you'd like to discuss before we go?",
    "[Neutral Facial Expression] [Slow Speech] I'm sorry to hear that you're experiencing pain in your knee. That must be quite uncomfortable. Physiotherapy can actually be really beneficial for addressing knee pain. The therapist can assess your condition and recommend exercises and techniques to help alleviate the discomfort and strengthen the muscles around your knee. By addressing the root cause of the pain, you can improve your mobility and overall quality of life. Would you like me to assist you in getting ready for your session?",
    "[Neutral Facial Expression] [Slow Speech] I'm sorry to hear that you're feeling under the weather. It's important to prioritize your health, especially when you're not feeling well. If you're experiencing flu-like symptoms, it might be best to rest and focus on getting better first. Physiotherapy can wait until you're feeling stronger and more able to participate fully. Would you like me to assist you in any way, such as getting some rest or arranging for any necessary medical assistance?"
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