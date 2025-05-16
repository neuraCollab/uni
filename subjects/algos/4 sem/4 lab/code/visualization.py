import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_methods_comparison(final_results, return_fig=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=final_results, x="Missing%", y="MeanRelativeError%", hue="Method", ax=ax)
    ax.set_title("Сравнение ошибок разных методов")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.show()


def plot_best_methods(best_methods, return_fig=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=best_methods, x="Missing%", y="MeanRelativeError%", hue="Method", ax=ax)
    ax.set_title("Лучшие методы по каждому уровню пропусков")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if return_fig:
        return fig, ax
    else:
        plt.tight_layout()
        plt.show()
