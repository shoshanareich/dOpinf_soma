import re


def as_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def config_list(config, *keys):
    for key in keys:
        if key in config:
            return as_list(config[key])
    return []


def format_tau(tau):
    return f"{tau:g}" if isinstance(tau, (int, float)) else str(tau)


def ensure_slash(path):
    return path if path.endswith("/") else path + "/"


def dirs_from_taus(prefix, taus):
    if not prefix or not taus:
        return []
    return [ensure_slash(f"{prefix}{format_tau(tau)}") for tau in taus]


def infer_tau_from_dir(path):
    match = re.search(r"tau([0-9]+(?:\.[0-9]+)?)", path)
    if not match:
        return None
    return float(match.group(1))


def resolve_forcing_config(config):
    train_taus = config_list(config, "train_taus", "tau_train", "train_tau", "train")
    test_taus = config_list(config, "test_taus", "tau_test", "test_tau", "test")

    snapshot_dirs = as_list(config.get("snapshot_dirs")) or dirs_from_taus(
        config.get("snapshot_dir_prefix") or config.get("train_dir_prefix"),
        train_taus,
    )
    test_dirs = as_list(config.get("test_dirs")) or as_list(config.get("test_dir")) or dirs_from_taus(
        config.get("test_dir_prefix") or config.get("snapshot_dir_prefix") or config.get("train_dir_prefix"),
        test_taus,
    )

    if not train_taus:
        train_taus = [infer_tau_from_dir(path) for path in snapshot_dirs]
    if not test_taus:
        test_taus = [infer_tau_from_dir(path) for path in test_dirs]

    return {
        "train_taus": train_taus,
        "test_taus": test_taus,
        "snapshot_dirs": snapshot_dirs,
        "test_dirs": test_dirs,
    }
