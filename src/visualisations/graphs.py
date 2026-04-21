import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os 
import glob
import pandas as pd

def afficher_apprentissage(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Évolution de l\'erreur pendant l\'entraînement')
    plt.xlabel('Époques')
    plt.ylabel('Erreur (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def affiche_prediction(df, prediction, valeur_de_travail):
    plt.figure(figsize=(12,6))
    plt.plot(df['time'], prediction, label="Prédiction")
    plt.plot(df['time'], df[valeur_de_travail], label="Mesure")
    plt.xlabel("Date")
    plt.ylabel(valeur_de_travail)
    plt.legend()
    plt.show()

def plot_scatter_zooms(error_df, processed_count):
    methods_unique = error_df['method'].unique()
    palette = sns.color_palette("tab10", n_colors=len(methods_unique))
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*']

    zooms = [1, 0.05, 0.005]

    for z_limit in zooms:
        plt.figure(figsize=(10, 6))

        for i, method in enumerate(methods_unique):
            sub = error_df[error_df['method'] == method]
            plt.scatter(sub['NMAE'], sub['NRMSE'],
                        color=palette[i%len(markers)],
                        marker=markers[i%len(markers)],
                        s=100,
                        edgecolor='k',
                        alpha=0.6,
                        label=method)

        plt.xlim(0, z_limit)
        plt.ylim(0, z_limit)
        plt.xlabel("NMAE")
        plt.ylabel("NRMSE")
        plt.title(f"Zoom [0 - {z_limit}] ({processed_count} fichiers)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_first_vs_last(stats_df, processed_count):
    plt.figure(figsize=(10, 7))

    methods = stats_df['method'].unique()
    palette = sns.color_palette("tab10", n_colors=len(methods))
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*']

    for i, method in enumerate(methods):
        row = stats_df[stats_df['method'] == method]

        x_val = row['Premier'].iloc[0]
        y_val = row['Dernier'].iloc[0]

        plt.scatter(
            x_val,
            y_val,
            color=palette[i],
            marker=markers[i% len(markers)],
            s=250,
            edgecolor='k',
            label=method
        )

        plt.text(x_val + 0.1, y_val + 0.1,
                 method,
                 fontsize=10,
                 fontweight='bold')

    plt.xlabel("Nombre de fois arrivée 1er")
    plt.ylabel("Nombre de fois arrivée dernier")
    plt.title(f"Fiabilité relative des méthodes\n(Total datasets : {processed_count})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_first_vs_avg_rank(stats_df, processed_count, val):
    plt.figure(figsize=(10, 7))

    methods = stats_df['method'].unique()
    palette = sns.color_palette("tab10", n_colors=len(methods))
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*']

    for i, method in enumerate(methods):
        row = stats_df[stats_df['method'] == method]

        x_val = row['Premier'].iloc[0]
        y_val = row['rank_moyen'].iloc[0]

        plt.scatter(
            x_val,
            y_val,
            color=palette[i],
            marker=markers[i % len(markers)],
            s=250,
            edgecolor='k',
            label=method
        )

        plt.text(x_val + 0.1, y_val + 0.02,
                 method,
                 fontsize=10,
                 fontweight='bold')

    plt.xlabel("Nombre de victoires (1er)")
    plt.ylabel("Rang moyen (plus petit = meilleur)")
    plt.title(f"Performance globale sur {val}\n({processed_count} fichiers)")

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dataset_focus(error_df, target_dataset_name):
    df_graph = error_df[error_df['dataset'] == target_dataset_name]

    if df_graph.empty:
        print(f"⚠️ Dataset {target_dataset_name} non trouvé.")
        return

    plt.figure(figsize=(12, 8))

    methods = df_graph['method'].unique()
    pct_values = sorted(df_graph['pct_removed'].unique())

    palette = sns.color_palette("husl", n_colors=len(methods))
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*']

    marker_map = {
        pct: markers[i % len(markers)]
        for i, pct in enumerate(pct_values)
    }

    for i, method in enumerate(methods):
        for pct in pct_values:
            subset = df_graph[
                (df_graph['method'] == method) &
                (df_graph['pct_removed'] == pct)
            ]

            if not subset.empty:
                plt.scatter(
                    subset['NRMSE'],
                    subset['NMAE'],
                    color=palette[i],
                    marker=marker_map[pct],
                    s=150,
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.9
                )

    legend_methods = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=m,
               markerfacecolor=palette[i],
               markersize=10)
        for i, m in enumerate(methods)
    ]

    legend_pct = [
        Line2D([0], [0],
               marker=marker_map[p],
               color='w',
               label=f"{int(p*100)}% trous",
               markerfacecolor='gray',
               markersize=10)
        for p in pct_values
    ]

    l1 = plt.legend(
        handles=legend_methods,
        title="Méthodes",
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    plt.gca().add_artist(l1)

    plt.legend(
        handles=legend_pct,
        title="% données retirées",
        loc='lower left',
        bbox_to_anchor=(1, 0)
    )

    plt.xlabel("NRMSE (sensibilité aux grosses erreurs)")
    plt.ylabel("NMAE (erreur moyenne globale)")
    plt.title(f"Analyse de robustesse : {target_dataset_name}")

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_error_heatmaps(error_df, error_metrics=['NMAE', 'NRMSE']):
    datasets = sorted(error_df['dataset'].unique())

    for metric in error_metrics:
        if metric not in error_df.columns:
            print(f"⚠️ Métrique {metric} absente dans le DataFrame.")
            continue

        # Pivot table : datasets en y, methods en x
        pivot_df = error_df.pivot_table(
            index='dataset',
            columns='method',
            values=metric,
            aggfunc='mean'  # moyenne si plusieurs pourcents retirés
        )

        plt.figure(figsize=(12, max(6, len(datasets)*0.3)))
        sns.heatmap(
            pivot_df,
            vmin=0,
            vmax=0.5,
            square=True,
            cbar_kws={'label': metric},
            linewidths=0
        )
        plt.title(f"Heatmap {metric} par dataset et méthode")
        plt.xlabel("Méthode")
        plt.ylabel("Dataset")
        plt.tight_layout()
        plt.show()

def plot_mean_errors_by_method(error_df, processed_count, valeur_de_travail):
    """
    Graphique montrant l'erreur moyenne (NMAE et NRMSE)
    pour chaque méthode sur l'ensemble des datasets.
    """

    mean_df = (
        error_df
        .groupby('method')[['NMAE', 'NRMSE']]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12,7), constrained_layout=True)

    methods = mean_df['method']
    palette = sns.color_palette("tab10", n_colors=len(methods))
    markers = ['o', '^', 's', 'D', 'v', 'P', 'X', '*']

    for i, row in mean_df.iterrows():
        plt.scatter(
            row['NMAE'],
            row['NRMSE'],
            color=palette[i],
            marker=markers[i % len(markers)],
            s=250,
            edgecolor='black'
        )

        plt.text(
            row['NMAE'],
            row['NRMSE'],
            row['method'],
            fontsize=11,
            fontweight='bold'
        )

    plt.xlabel("NMAE moyen")
    plt.ylabel("NRMSE moyen")
    plt.title(f"Erreur moyenne par méthode sur {valeur_de_travail}\n({processed_count} datasets)")

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

def plot_nappes(dossier, valeur_de_travail="niveau_nappe_eau", valeur_de_travail2=None):
    fichiers = glob.glob(os.path.join(dossier, "*.csv"))

    for fichier in fichiers:
        try:
            df = pd.read_csv(fichier, sep=";")

            # Conversion du temps
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')

            # Conversion de la valeur
            df[valeur_de_travail] = pd.to_numeric(df[valeur_de_travail], errors='coerce')

            # Plot
            plt.figure(figsize=(10,5))
            plt.plot(df['time'], df[valeur_de_travail])
            if valeur_de_travail2:
                plt.plot(df['time'], df[valeur_de_travail2])
            plt.title(f"Nappe : {os.path.basename(fichier)}")
            plt.xlabel("Temps")
            plt.ylabel(valeur_de_travail)
            plt.grid(True)

            plt.show()

        except Exception as e:
            print(f"Erreur avec {fichier} : {e}")