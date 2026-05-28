import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "sensing":          os.path.join(BASE_DIR, "sensings", "sensing.csv"),
    "steps":            os.path.join(BASE_DIR, "sensings", "steps.csv"),
    "data_dict_daily":  os.path.join(BASE_DIR, "sensings", "Data Dictionary (Daily).csv"),
    "data_dict_hourly": os.path.join(BASE_DIR, "sensings", "Data Dictionary (Hourly).csv"),
    "reports":          os.path.join(BASE_DIR, "reports"),
    "cleaned_data":     os.path.join(BASE_DIR, "cleaned_data"),
}

# Zaman dilimleri (sensing.csv ep kodları)
EPISODES = {
    "ep_0": "full_day",
    "ep_1": "midnight_to_9am",
    "ep_2": "9am_to_6pm",
    "ep_3": "6pm_to_midnight",
}

# Sensör sütun grupları (prefix bazlı)
COLUMN_GROUPS = {
    "activity":  ["act_"],
    "audio":     ["audio_"],
    "location":  ["loc_"],
    "call":      ["call_"],
    "sms":       ["sms_"],
    "unlock":    ["unlock_"],
    "steps":     ["step_"],
    "sleep":     ["sleep_"],
    "light":     ["light_"],
    "other_app": ["other_playing_"],
    "quality":   ["quality_"],
}

# Sayısal sütunlarda kabul edilebilir minimum değer (negatif olamaz)
NON_NEGATIVE_PREFIXES = [
    "act_", "audio_", "loc_", "call_", "sms_",
    "unlock_", "step_", "sleep_duration", "light_",
]

# Modelin lag1/rmean7 ürettiği feature'lar (server.py runtime'da kullanır)
LAG_FEATURES = [
    "unlock_num_ep_0",
    "unlock_duration_ep_0",
    "gunduz_gece_telefon_orani",
    "sedanter_saat",
    "aktivite_toplam",
    "act_still_ep_0",
    "mobilite_skoru",
    "loc_dist_ep_0",
    "loc_max_dis_from_campus_ep_0",
    "audio_amp_mean_ep_0",
    "audio_convo_duration_ep_0",
    "sosyal_iletisim_yogunluk",
]

# Modelin rstd7 (7 günlük std) ürettiği feature'lar
ROLLING_STD_FEATURES = [
    "unlock_num_ep_0",
    "sedanter_saat",
    "aktivite_toplam",
    "mobilite_skoru",
    "audio_amp_mean_ep_0",
    "gunduz_gece_telefon_orani",
]
