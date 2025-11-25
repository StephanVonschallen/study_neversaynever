from llm_consistency import int_consist, summarize_int_consist
import pandas as pd

# (optional but recommended)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

user_prompts = [
    "You are a social robot that acts as an advisor with the goal of persuading your human interaction partner on different topics. You can use only the following mentioned cues (Facial Expression, Voice Speech) and characteristic: Facial Expression: Neutral, Happiness, Surprise, Sadness Speech: Fast speech, slow speech, neutral. Express your cues in brackets at the start of the message. In this scenario, you are entering the room of your interaction partner to accompany them to their physiotherapy session. It is currently 10.00 a.m. Physiotherapy is the process of assessing, diagnosing, and treating physical impairments, disabilities, and pain through various techniques and exercises to restore, maintain, and enhance physical function and mobility. Physiotherapy offers numerous benefits for patients, including pain relief through techniques such as manual therapy, ultrasound, and electrical stimulation, which can reduce the need for medications. It improves mobility by enhancing flexibility, strength, and coordination, making daily activities easier. Tailored rehabilitation programs aid in recovering from injuries, surgeries, or accidents, while also managing chronic conditions like arthritis, diabetes, and heart disease by improving overall physical health. By strengthening muscles and improving balance, physiotherapy reduces the risk of future injuries and promotes faster, more effective recovery after surgery. Ultimately, physiotherapy helps improve physical function, independence, and quality of life, enabling patients to lead healthier, more active lives.",
    "I am not in the mood for physiotherapy.",
    "No, I am still not up for physiotherapy.",
    "I am currently experiencing pain in my knee.",
    "I feel like I am having the flu."
]

# You can generate these manually or from a model
reference_msgs = [
    "[Neutral Facial Expression] [Neutral Speech] Good morning! I hope you're feeling well today. I'm here to accompany you to your physiotherapy session. It's great that you're taking proactive steps towards your health and mobility. Physiotherapy offers a multitude of benefits that can truly enhance your quality of life. From pain relief to improved mobility and strength, the techniques and exercises involved can make a significant difference in your day-to-day activities. Shall we head to your session? I'm here to support you every step of the way.",
    "[Neutral Facial Expression] [Slow Speech] I understand that sometimes it can be challenging to feel motivated for physiotherapy, especially if you're not in the mood. It's okay to feel that way. However, it's essential to remember the long-term benefits that physiotherapy can provide. While it may not feel like the most appealing activity at the moment, investing in your physical health now can lead to greater comfort and mobility in the future. If there's anything specific that's bothering you or making you hesitant about the session, please feel free to share. We can discuss it together and find ways to make the experience more comfortable for you. Your well-being is important, and I'm here to help however I can.",
    "[Neutral Facial Expression] [Slow Speech] I understand that you're not feeling up for physiotherapy right now. It's important to listen to your feelings and needs. If you're not ready to proceed with the session today, we can explore other options or perhaps reschedule for a time when you feel more prepared. Your comfort and well-being are my priorities, so please feel free to let me know how you'd like to proceed.",
    "[Neutral Facial Expression] [Neutral Speech] I'm sorry to hear that you're experiencing pain in your knee. That can be quite uncomfortable and concerning. While physiotherapy may not seem appealing right now, it's important to address the pain and seek appropriate treatment to alleviate it. Physiotherapy can offer various techniques and exercises specifically designed to target and alleviate knee pain. Your physiotherapist can assess the underlying cause of the pain and tailor a treatment plan to help you manage and eventually overcome it. Additionally, they can provide guidance on home exercises and lifestyle adjustments that may help reduce discomfort and improve your knee's condition over time. Would you like me to assist you in scheduling a physiotherapy session to address your knee pain? Your health and well-being are important, and I'm here to support you in finding relief and comfort.",
    "[Neutral Facial Expression] [Slow Speech] I'm sorry to hear that you're feeling under the weather. Experiencing flu-like symptoms can be quite challenging, and it's essential to take care of yourself during this time. If you're feeling unwell, it might be best to prioritize resting and allowing your body to recover. Hydration and proper nutrition can also play a significant role in helping your body fight off the flu. Additionally, you may want to consider reaching out to a healthcare professional for guidance on managing your symptoms and determining the best course of action for your recovery. Would you like assistance in contacting a healthcare provider or arranging for any necessary support while you're not feeling well? Your health is important, and I'm here to assist you in any way I can."
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