import matplotlib.pyplot as plt
import os

def customize_save_and_show_plot(plot_name, current_dir):
    # Turning on grid lines to help in visualizing data points
    plt.grid(True, which='both', linewidth=0.5)
    # plt.grid(True, which='both')

    # Making the border (spine) of the plot less prominent and gray
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(colors='gray', labelsize=16)
    ax.yaxis.get_label().set_fontsize(18)
    ax.xaxis.get_label().set_fontsize(18)

    title = ax.title  # Get the title object
    title.set_fontsize(20)  # Set the font size

    # Adjusting the layout to prevent overlap and ensure everything fits well within the plot area
    plt.tight_layout()

    # Saving the plot to the 'plots' folder within the 'experiment' folder
    current_dir = os.getcwd()  # Get current working directory
    experiment_dir = os.path.join(current_dir, 'data', 'plots')
    os.makedirs(experiment_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plot_path = os.path.join(experiment_dir, f"{plot_name}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Displaying the plot
    plt.show()
