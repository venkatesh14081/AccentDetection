import random

# ================= AUSTRALIAN =================
AUSTRALIAN_SENTENCES = [
    ("I wake up early every day.", "s01.wav"),
    ("She is preparing for her exams.", "s02.wav"),
    ("He likes to play cricket in the evening.", "s03.wav"),
    ("They are waiting for the bus.", "s04.wav"),
    ("I am learning English for my career.", "s05.wav"),
    ("The sun rises in the east.", "s06.wav"),
    ("She completed her work on time.", "s07.wav"),
    ("We are planning to attend the interview.", "s08.wav"),
    ("He drinks water after exercise.", "s09.wav"),
    ("The train arrived late today.", "s10.wav"),
    ("I practice coding every night.", "s11.wav"),
    ("She speaks politely to everyone.", "s12.wav"),
    ("They finished the project successfully.", "s13.wav"),
    ("He is confident about his future.", "s14.wav"),
    ("I read newspapers daily.", "s15.wav"),
    ("The teacher explained the topic clearly.", "s16.wav"),
    ("We should manage time properly.", "s17.wav"),
    ("She helps her friends in need.", "s18.wav"),
    ("He is improving his communication skills.", "s19.wav"),
    ("I believe hard work brings success.", "s20.wav"),
]

# ================= OTHER ACCENTS =================
NORMAL_SENTENCES = {
    "american": [
        ("Could you please give me some time to analyze this properly before responding?", "s01.wav"),
        ("I believe that small daily efforts will eventually lead to big changes in my life.", "s02.wav"),
        ("No matter what happens, I choose to stay calm and move forward with confidence.", "s03.wav")
    ],

    "british": [
        ("Could you please give me some time to analyze this properly before responding?", "s01.wav"),
        ("I believe that small daily efforts will eventually lead to big changes in my life.", "s02.wav"),
        ("No matter what happens, I choose to stay calm and move forward with confidence.", "s03.wav")
    ],

    "canadian": [
        ("Could you please give me some time to analyze this properly before responding?", "s01.wav"),
        ("I believe that small daily efforts will eventually lead to big changes in my life.", "s02.wav"),
        ("No matter what happens, I choose to stay calm and move forward with confidence.", "s03.wav")
    ]
}

# ================= MAIN FUNCTION =================
def get_random_sentence(accent: str):
    accent = accent.lower()

    if accent == "australian":
        return random.choice(AUSTRALIAN_SENTENCES)

    elif accent in NORMAL_SENTENCES:
        return random.choice(NORMAL_SENTENCES[accent])

    else:
        return ("Please select a valid accent.", None)