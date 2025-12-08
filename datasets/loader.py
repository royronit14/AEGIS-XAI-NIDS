from .unsw_loader import load_unsw
from .cicids_loader import load_cicids
from .botiot_loader import load_botiot

def load_dataset(name):
    name = name.lower()

    if name in ["unsw", "unsw-nb15"]:
        return load_unsw()

    elif name in ["cicids", "cicids2017"]:
        return load_cicids()

    elif name in ["botiot", "bot-iot"]:
        return load_botiot()

    else:
        raise ValueError(f"Unknown dataset: {name}")
