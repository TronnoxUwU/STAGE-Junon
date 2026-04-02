import csv
import itertools
import pathlib
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from methodes import cnn_model, lstm_model, bilstm_model, fit

# ─────────────────────────────────────────────
#  Hyperparamètres à explorer
# ─────────────────────────────────────────────
learning_rates = [1e-4, 1e-3]
weight_decays  = [0.0, 1e-5, 1e-4]
n_units_list   = [32, 64]
dropouts       = [0.2, 0.3]

models_dict = {
    "cnn":    cnn_model,
    "lstm":   lstm_model,
    "bilstm": bilstm_model,
}

# ─────────────────────────────────────────────
#  Fichier de log CSV  (thread-safe via lock)
# ─────────────────────────────────────────────
LOG_FILE   = pathlib.Path("grid_search_log.csv")
CSV_FIELDS = ["model", "lr", "wd", "units", "dropout", "val_loss", "val_accuracy"]
_csv_lock  = threading.Lock()


def _log_result(result: dict) -> None:
    """Écrit un résultat dans le CSV (thread-safe)."""
    with _csv_lock:
        write_header = not LOG_FILE.exists()
        with LOG_FILE.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: result[k] for k in CSV_FIELDS})


def _load_done_configs() -> set:
    """Lit le CSV existant et renvoie l'ensemble des configs déjà testées."""
    done = set()
    if LOG_FILE.exists():
        with LOG_FILE.open(newline="") as f:
            for row in csv.DictReader(f):
                done.add((
                    row["model"],
                    float(row["lr"]),
                    float(row["wd"]),
                    int(row["units"]),
                    float(row["dropout"]),
                ))
    return done


# ─────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────
def _train_one(model_name, model_fn, lr, wd, n_units, dropout,
               X_train, y_train, X_val, y_val) -> dict:
    """Entraîne un modèle pour une config donnée."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_fn(
        input_shape=X_train,
        learning_rate=lr,
        weight_decay=wd,
        n_units=n_units,
        dropout=dropout,
    ).to(device)

    history = fit(model, X_train, y_train, X_val, y_val)

    val_loss     = float(min(history.history["val_loss"]))
    val_accuracy = float(max(history.history.get("val_accuracy", [float("nan")])))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        gc.collect()

    return {
        "model":        model_name,
        "lr":           lr,
        "wd":           wd,
        "units":        n_units,
        "dropout":      dropout,
        "val_loss":     val_loss,
        "val_accuracy": val_accuracy,
        "_model_obj":   model,
    }


# ─────────────────────────────────────────────
#  Grid search principal
# ─────────────────────────────────────────────
def grid_search_all(X_train, y_train, X_val, y_val, n_workers: int = 3):
    """
    Lance un grid search parallèle (threads) avec :
      - reprise automatique si le CSV existe déjà,
      - logging immédiat après chaque run (thread-safe),
      - sélection finale sur val_loss (+ val_accuracy en tie-break).

    n_workers : nombre de modèles entraînés en parallèle.
                Avec 1 GPU, 2-4 est un bon point de départ selon la taille
                des modèles. Augmenter si le GPU n'est pas saturé (< 80 %),
                diminuer si OOM.
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    done_configs = _load_done_configs()
    if done_configs:
        print(f"↩️  Reprise : {len(done_configs)} config(s) déjà terminée(s), ignorées.")

    # ── Construction de la liste de configs ──────────────────────────────────
    all_configs = []
    for model_name, model_fn in models_dict.items():
        for lr, wd, n_units, dropout in itertools.product(
            learning_rates, weight_decays, n_units_list, dropouts
        ):
            key = (model_name, lr, wd, n_units, dropout)
            if key not in done_configs:
                all_configs.append((model_name, model_fn, lr, wd, n_units, dropout))

    total     = len(all_configs) + len(done_configs)
    remaining = len(all_configs)
    print(f"🔍 {remaining} config(s) à entraîner sur {total} au total.")
    print(f"⚡ Parallélisation : {n_workers} workers (threads)\n")

    if not remaining:
        print("✅ Toutes les configs ont déjà été testées. Lecture du CSV.")
        return _best_from_csv()

    # ── Boucle parallèle ─────────────────────────────────────────────────────
    results_raw = []
    pbar = tqdm(total=remaining, desc="Grid search") if _tqdm_available else None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _train_one,
                model_name, model_fn, lr, wd, n_units, dropout,
                X_train, y_train, X_val, y_val,
            ): (model_name, lr, wd, n_units, dropout)
            for model_name, model_fn, lr, wd, n_units, dropout in all_configs
        }

        for future in as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
            except Exception as e:
                model_name, lr, wd, n_units, dropout = cfg
                print(f"\n❌ Échec — {model_name} lr={lr} wd={wd} "
                      f"units={n_units} dropout={dropout} : {e}")
                if pbar:
                    pbar.update(1)
                continue

            _log_result(result)
            results_raw.append(result)

            if pbar:
                pbar.set_postfix({
                    "model": result["model"],
                    "loss":  f"{result['val_loss']:.4f}",
                })
                pbar.update(1)
            else:
                done_so_far = len(done_configs) + len(results_raw)
                print(f"[{done_so_far}/{total}] {result['model']} "
                      f"lr={result['lr']} wd={result['wd']} "
                      f"units={result['units']} dropout={result['dropout']} "
                      f"→ loss={result['val_loss']:.4f}  "
                      f"acc={result['val_accuracy']:.4f}")

    if pbar:
        pbar.close()

    if not results_raw:
        print("⚠️  Aucun résultat collecté (tous en échec ?).")
        return None, {}, []

    # ── Sélection du meilleur ────────────────────────────────────────────────
    best = min(results_raw, key=lambda r: (r["val_loss"], -r["val_accuracy"]))

    best_model  = best["_model_obj"]
    best_config = {k: best[k] for k in ["model", "lr", "wd", "units", "dropout"]}

    print("\n🏆 BEST CONFIG:")
    for k, v in best_config.items():
        print(f"   {k}: {v}")
    print(f"🏆 val_loss     : {best['val_loss']:.4f}")
    print(f"🏆 val_accuracy : {best['val_accuracy']:.4f}")
    print(f"\n📄 Résultats sauvegardés dans : {LOG_FILE.resolve()}")

    results = [{k: r[k] for k in CSV_FIELDS} for r in results_raw]
    return best_model, best_config, results


# ─────────────────────────────────────────────
#  Utilitaire : relire le meilleur depuis le CSV
# ─────────────────────────────────────────────
def _best_from_csv():
    rows = []
    with LOG_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "model":        row["model"],
                "lr":           float(row["lr"]),
                "wd":           float(row["wd"]),
                "units":        int(row["units"]),
                "dropout":      float(row["dropout"]),
                "val_loss":     float(row["val_loss"]),
                "val_accuracy": float(row["val_accuracy"]),
            })

    best = min(rows, key=lambda r: (r["val_loss"], -r["val_accuracy"]))
    print("\n🏆 BEST CONFIG (depuis CSV):")
    for k, v in best.items():
        print(f"   {k}: {v}")

    return None, {k: best[k] for k in ["model", "lr", "wd", "units", "dropout"]}, rows