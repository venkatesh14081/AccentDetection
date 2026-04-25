def pronunciation_feedback(accent, spoken_text, sentence):
    sentence_words = sentence.lower().replace(".", "").split()

    # ---------------- WORD STRESS (Sentence-specific) ----------------
    stress_words = [
        w for w in sentence_words
        if len(w) >= 5 and w not in ["should", "because", "through"]
    ]

    # ---------------- PAUSES ----------------
    pauses = []

    if "," in sentence:
        pauses.append("Pause briefly at commas")

    if "and" in sentence_words:
        pauses.append("Short pause before 'and'")

    if "but" in sentence_words:
        pauses.append("Pause before 'but'")

    # ---------------- SPEAKING FLOW ----------------
    flow_map = {
        "american": "Strong stress on key words with confident pace",
        "british": "Steady rhythm with controlled pauses",
        "australian": "Relaxed tone with smooth connected speech",
        "canadian": "Neutral flow with balanced stress"
    }

    # ---------------- ACCENT PRONUNCIATION RULES ----------------
    accent_rules = {
        "american": [
            "Pronounce 'r' sounds clearly",
            "Use strong vowel sounds"
        ],
        "british": [
            "Soften 'r' sounds at word endings",
            "Keep vowels crisp"
        ],
        "australian": [
            "Relax vowels slightly",
            "Avoid sharp endings of words"
        ],
        "canadian": [
            "Neutral vowel sounds",
            "Clear but soft consonants"
        ]
    }

    feedback = []

    feedback.append(
        f"🔸 **Word Stress:** Emphasize → {', '.join(stress_words)}"
    )

    if pauses:
        feedback.append(
            f"⏸️ **Pauses:** {' | '.join(pauses)}"
        )

    feedback.append(
        f"🎵 **Speaking Flow:** {flow_map.get(accent)}"
    )

    feedback.append(
        f"🗣️ **Accent Tips:** " + " | ".join(accent_rules.get(accent))
    )

    return feedback
