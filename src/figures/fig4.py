import matplotlib.pyplot as plt
from pathlib import Path

image_folder = Path("images", "fig4")

if __name__ == '__main__':
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # load image from images fig2 folder
    ae_workflow = plt.imread(Path("images", "fig4", "ae_workflow.png"))

    fig = plt.figure(figsize=(5, 2), dpi=300)
    gspec = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(gspec[:, :])
    # remove box from ax1
    plt.box(False)
    # remove ticks from ax1
    ax1.set_xticks([])
    ax1.set_yticks([])
    # show spatial information image
    ax1.imshow(ae_workflow, aspect='auto')
    plt.tight_layout()
    plt.tight_layout()

    # save figure
    fig.savefig(Path(image_folder, "fig3a.png"), dpi=300, bbox_inches='tight')
    fig.savefig(Path(image_folder, "fig3a.eps"), dpi=300, bbox_inches='tight', format='eps')
