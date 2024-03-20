import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from collections import defaultdict

def culmulative_return_calculate(return_array : np):
    Culmulative_Return = 0
    Culmulative_Return_np = np.zeros(len(return_array))

    for i in range(len(return_array)):
        Culmulative_Return += return_array[i]
        Culmulative_Return_np[i] = Culmulative_Return
    return Culmulative_Return_np

def reward_count(n_episodes, n_chunks, data : np):
    # Size of each chunk
    chunk_size = n_episodes // n_chunks

    positive_count_ls = []
    negative_count_ls = []

    print("====================== Reward Frequency ======================\n")

    # Plot
    categories = []
    for i in range(n_chunks):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk_data = data[start_index:end_index]
        categories.append(end_index)
        positive_reward = np.count_nonzero(chunk_data == 1)
        negative_reward = np.count_nonzero(chunk_data == -1)
        positive_count_ls.append(positive_reward)
        negative_count_ls.append(negative_reward)

        positive_reward_percent = positive_reward/len(chunk_data)
        negative_reward_percent = negative_reward/len(chunk_data)
        print("Iteration[{}:{}]     positive:{} negative:{}".format(start_index, end_index, positive_reward_percent, negative_reward_percent))

    categories_list = list(map(str, categories))
    
    print("\n====================== Reward Frequency ======================")

    return categories_list, positive_count_ls, negative_count_ls

def reward_plot(n_episodes, n_chunks, return_queue):
    return_array = np.array(return_queue).flatten()
    Culmulative_Return_np = culmulative_return_calculate(return_array)
    categories_list, positive_count_ls, negative_count_ls = reward_count(n_episodes, n_chunks, return_array)

    fig, axs = plt.subplots(ncols=3, figsize=(20, 8))
    axs[0].set_title("Cumulative return Plot")
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Cumulative return')
    axs[0].plot(range(len(Culmulative_Return_np)), Culmulative_Return_np)
    axs[0].grid(True)

    rolling_length = 500
    axs[1].set_title("Episode Return Filtered Plot")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            return_array, np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[1].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].grid(True)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Filter Return')

    axs[2].bar(categories_list, positive_count_ls, label='-1')
    axs[2].bar(categories_list, negative_count_ls, bottom=positive_count_ls, label='1')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Reward Frequency')
    axs[2].set_title(f'Reward Frequency Bar Chart')

    plt.tight_layout()
    plt.legend()
    plt.show()

def training_plot(return_queue, length_queue, error_queue, rolling_length):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(error_queue), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

def create_grids(item, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in item:
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig