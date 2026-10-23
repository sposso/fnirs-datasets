"""Loaders of the fNIRS pretraining corpus and of the downstream datasets.

Each loader returns ``{recording_id: record}``; see the README for the fields of a record. Modules are imported
only when a dataset is loaded, so a dataset's optional dependencies are needed only for that dataset.

    from loaders import load
    records = load("fresh_motor")
"""

import importlib

PRETRAINING = {  # dataset name -> "module:function"
    "ai_persuasion":         "load_ai_persuasion_data:load_ai_persuasion_data",
    "alice":                 "load_alice_data:load_alice_data",
    "anesthesia":            "load_anesthesia_data:load_anesthesia_data",
    "appearance":            "load_appearance_data:load_appearance_data",
    "athlete":               "load_athlete_data:load_athlete_data",
    "attention_tasks":       "load_attention_tasks_data:load_attention_tasks_data",
    "audio_fresh":           "load_audio_fresh:load_fresh_audio_data",
    "audio_visual_speech":   "load_audio_visual_speech:load_audio_visual_speech_data",
    "baby":                  "load_baby_data:load_baby_data",
    "balance_tasks":         "load_balance_tasks:load_balance_tasks_data",
    "ball_squeezing":        "load_ball_squeezing:load_ball_squeezing_data",
    "cambridge_hddot":       "load_cambridge_hddot:load_cambridge_hddot_data",
    "cambridge_visual":      "load_cambridge_visual:load_cambridge_visual_data",
    "cmx7":                  "load_cmx7_data:load_cmx7_data",
    "cocktail_party":        "load_cocktail_party:load_cocktail_party_data",
    "coop_visuomotor":       "load_coop_visuomotor_data:load_coop_visuomotor_data",
    "cosmetic":              "load_cosmetic_data:load_cosmetic_data",
    "domribrain":            "load_domribrain_data:load_domribrain_data",
    "driving":               "load_driving_data:load_driving_data",
    "emotion_reappraisal":   "load_emotion_reappraisal_data:load_emotion_reappraisal_data",
    "finemi":                "load_finemi_data:load_finemi_data",
    "finger_leg":            "load_finger_leg_data:load_finger_leg_data",
    "finger_movements":      "load_finger_movements:load_finger_movements_data",
    "flight":                "load_flight_data:load_flight_data",
    "fog_parkinson":         "load_fog_parkinson_data:load_fog_parkinson_data",
    "fresh_motor":           "load_fresh_motor:load_fresh_motor_data",
    "hdot_adults_children":  "load_hdot_adults_children:load_hdot_adults_children_data",
    "hdot_validation":       "load_hdot_validation:load_hdot_validation_data",
    "hyperscanning":         "load_hyperscanning:load_hyperscanning_data",
    "inclusion":             "load_inclusion_data:load_inclusion_data",
    "interaural_coherence":  "load_interaural_coherence_data:load_interaural_coherence_data",
    "intergenerational":     "load_intergenerational_data:load_intergenerational_data",
    "language_production":   "load_language_production_data:load_language_production_data",
    "lexical_retest":        "load_lexical_retest_data:load_lexical_retest_data",
    "luke_tapping":          "load_luke_tapping:load_luke_tapping_data",
    "mind":                  "load_mind_data:load_mind_data",
    "morphology":            "load_morphology_data:load_morphology_data",
    "multitask_error":       "load_multitask_error_data:load_multitask_error_data",
    "parents":               "load_parents_data:load_parents_data",
    "physionet_nback":       "load_physionet_nback:load_physionet_nback_data",
    "pilot_workload":        "load_pilot_workload_data:load_pilot_workload_data",
    "pipa_music":            "load_pipa_music_data:load_pipa_music_data",
    "prosody":               "load_prosody_data:load_prosody_data",
    "refed":                 "load_refed_data:load_refed_data",
    "resident_workload":     "load_resident_workload:load_resident_workload_data",
    "rumor":                 "load_rumor_data:load_rumor_data",
    "shin2016":              "load_shin2016:load_shin2016_data",
    "shin2018":              "load_shin2018:load_shin2018_data",
    "social":                "load_social_data:load_social_data",
    "spouse":                "load_spouse_data:load_spouse_data",
    "stroke_bci":            "load_stroke_bci:load_stroke_bci_data",
    "stroke_hddot":          "load_stroke_hddot:load_stroke_hddot_data",
    "stroop":                "load_stroop_data:load_stroop_data",
    "surgical":              "load_surgical_data:load_surgical_data",
    "symmetry_wm":           "load_symmetry_wm_data:load_symmetry_wm_data",
    "turkish_relative":      "load_turkish_relative_data:load_turkish_relative_data",
    "vhddot_semantic_movies":"load_vhddot_semantic_movies:load_vhddot_semantic_movies_data",
    "visuomotor":            "load_visuomotor_data:load_visuomotor_data",
    "voice_pitch":           "load_voice_pitch_data:load_voice_pitch_data",
    "walking":               "load_walking_data:load_walking_data",
    "yoga":                  "load_yoga_data:load_yoga_data",
}

DOWNSTREAM = {
    "hefmi_ich":             "downstream.load_hefmi_ich:load_hefmi_ich_data",
    "mirror":                "downstream.load_mirror:load_mirror_data",
    "speech_comprehension":  "downstream.load_speech_comprehension:load_speech_comprehension_data",
    "stroke_recovery":       "downstream.load_stroke_recovery:load_stroke_recovery_data",
    "tinnitus":              "downstream.load_tinnitus_data:load_tinnitus_data",
    "tools_animals":         "downstream.load_tools_animals:load_tools_animals_data",
}


def load(name, **kwargs):
    """Load one dataset of ``PRETRAINING`` or ``DOWNSTREAM`` by name; ``kwargs`` go to its loader."""
    module, function = {**PRETRAINING, **DOWNSTREAM}[name].split(":")
    return getattr(importlib.import_module(f"loaders.{module}"), function)(**kwargs)
