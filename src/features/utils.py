from typing import Dict


def build_category_maps():
    category_map: Dict[str, int] = {
        "groceries": 0,
        "electronics": 1,
        "travel": 2,
        "restaurants": 3,
        "utilities": 4,
        "others": 5,
    }

    device_type_map: Dict[str, int] = {
        "mobile": 0,
        "desktop": 1,
        "pos_terminal": 2,
    }

    channel_map: Dict[str, int] = {
        "online": 0,
        "in_store": 1,
    }

    entry_mode_map: Dict[str, int] = {
        "chip": 0,
        "swipe": 1,
        "contactless": 2,
        "manual": 3,
        "online": 4,
    }

    country_map: Dict[str, int] = {
        "IN": 0,
        "US": 1,
        "GB": 2,
        "AE": 3,
        "SG": 4,
    }

    return {
        "category_map": category_map,
        "device_type_map": device_type_map,
        "channel_map": channel_map,
        "entry_mode_map": entry_mode_map,
        "country_map": country_map,
    }
