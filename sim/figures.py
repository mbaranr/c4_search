import sys
import numpy as np
import pandas as pd
import seaborn as sns
from c4.state import C4State
import matplotlib.pyplot as plt
from search.node import NodeMCTS, NodeMinimax

def fig_1(path_to_table: str, out_path: str):

    # first curate the data

    sims = pd.read_parquet(path_to_table)

    sims['adjusted_win'] = ((sims['agent_curr'] == 'minimax') & sims['is_win']) | (
        (sims['agent_curr'] == 'mcts') & sims['budget_exceeded']
    )

    outcomes_ = sims.groupby('sim_id')['adjusted_win'].any().reset_index()
    metadata = sims[['sim_id', 'depth', 'budget_total', 'mcts_strategy']].drop_duplicates()
    outcomes = pd.merge(outcomes_, metadata, on='sim_id')

    # line plots
    line_grouped = outcomes.groupby(['depth', 'budget_total'])['adjusted_win'].mean().reset_index()
    line_pivoted = line_grouped.pivot(index='depth', columns='budget_total', values='adjusted_win')

    # heatmaps
    heatmap_grouped = outcomes.groupby(['depth', 'budget_total', 'mcts_strategy'])['adjusted_win'].mean().reset_index()
    strategies = heatmap_grouped['mcts_strategy'].unique()

    # minimax losses
    minimax_losses = sims[(sims['agent_curr'] == 'minimax') & (sims['budget_exceeded'])]
    minimax_loss_counts = minimax_losses.groupby(['depth', 'budget_total']).size().reset_index(name='losses')

    # mcts losses
    mcts_losses = sims[(sims['agent_curr'] == 'mcts') & (sims['budget_exceeded'])]
    mcts_loss_counts = mcts_losses.groupby(['mcts_strategy', 'budget_total']).size().reset_index(name='losses')

    budgets = minimax_loss_counts['budget_total'].unique()
    depths = minimax_loss_counts['depth'].unique()
    strats = mcts_loss_counts['mcts_strategy'].unique()

    full_combinations_mm = pd.DataFrame([(depth, budget) for depth in depths for budget in budgets], columns=['depth', 'budget_total'])
    full_combinations_mcts = pd.DataFrame([(strat, budget) for strat in strats for budget in budgets], columns=['mcts_strategy', 'budget_total'])

    minimax_loss_counts = pd.merge(full_combinations_mm, minimax_loss_counts, how='left', on=['depth', 'budget_total'])
    mcts_loss_counts = pd.merge(full_combinations_mcts, mcts_loss_counts, how='left', on=['mcts_strategy', 'budget_total'])

    minimax_loss_counts['losses'].fillna(0, inplace=True)
    mcts_loss_counts['losses'].fillna(0, inplace=True)

    # now plot the data

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))  # 2 rows, 3 columns 

    # line plot (top-left)
    ax_line = axes[0, 0]
    for budget in line_pivoted.columns:
        ax_line.plot(
            line_pivoted.index, line_pivoted[budget], marker='o', label=budget
        )
    ax_line.set_title('Depth on Minimax Performance', fontsize=16)
    ax_line.set_xlabel('Depth', fontsize=14)
    ax_line.set_ylabel('Win Rate', fontsize=14)
    ax_line.set_xticks(line_pivoted.index)
    ax_line.legend(title='Budget', fontsize=10)
    ax_line.grid(alpha=0.5)

    # bar plot (top-center)
    ax_top_center = axes[0, 1]
    width = 0.15
    x1 = np.arange(len(depths))
    for i, budget in enumerate(budgets):
        budget_data = minimax_loss_counts[minimax_loss_counts['budget_total'] == budget]
        ax_top_center.bar(x1 + (i - 2) * width, budget_data['losses'], width=width, label=budget)
    ax_top_center.set_title('Losses Due to Budget Exceeded: Minimax', fontsize=16)
    ax_top_center.set_xlabel('Depth', fontsize=14)
    ax_top_center.set_ylabel('Number of Losses', fontsize=14)
    ax_top_center.set_xticks(x1)
    ax_top_center.set_xticklabels(depths, fontsize=12)
    ax_top_center.grid(True, axis='y', alpha=0.5)

    # bar plot (top-right)
    ax_top_right = axes[0, 2]
    x2 = np.arange(len(strats))
    for i, budget in enumerate(budgets):
        budget_data = mcts_loss_counts[mcts_loss_counts['budget_total'] == budget]
        ax_top_right.bar(x2 + (i - 2) * width, budget_data['losses'], width=width, label=budget)
    ax_top_right.set_title('Losses Due to Budget Exceeded: MCTS', fontsize=16)
    ax_top_right.set_xlabel('MCTS Strategy', fontsize=14)
    ax_top_right.set_ylabel('Number of Losses', fontsize=14)
    ax_top_right.set_xticks(x2)
    ax_top_right.set_xticklabels(strats, fontsize=12)
    ax_top_right.grid(True, axis='y', alpha=0.5)

    # heat maps (bottom row)
    vmin, vmax = heatmap_grouped['adjusted_win'].min(), heatmap_grouped['adjusted_win'].max()
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.25])  

    for i, strategy in enumerate(strategies):
        ax_bottom = axes[1, i]  
        strategy_data = heatmap_grouped[heatmap_grouped['mcts_strategy'] == strategy]
        pivoted = strategy_data.pivot(index='depth', columns='budget_total', values='adjusted_win')

        sns.heatmap(pivoted, ax=ax_bottom, annot=True, fmt=".2f", cmap="viridis", cbar=i == 0,
                    cbar_ax=cbar_ax if i == 0 else None, vmin=vmin, vmax=vmax, linewidths=0.5)
        
        ax_bottom.set_title(f"MCTS Strategy: {strategy}", fontsize=12)
        ax_bottom.set_xlabel("Budget", fontsize=10)
        ax_bottom.set_ylabel("Depth" if i == 0 else "", fontsize=10)

    cbar = fig.colorbar(ax_bottom.collections[0], cax=cbar_ax)  # colorbar only for last heatmap
    cbar.set_label('Win Rate', fontsize=12)
    cbar.outline.set_edgecolor('black')  
    cbar.outline.set_linewidth(0.5)  #

    fig.text(0.01, 1, 'a', ha='center', fontsize=18, weight='bold')
    fig.text(0.01, 0.5, 'c', ha='center', fontsize=18, weight='bold')
    fig.text(0.32, 1, 'b', ha='center', fontsize=18, weight='bold')
    fig.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.savefig(out_path, dpi=300, bbox_inches='tight')


def fig_2(path_to_table: str, best_depths: dict, out_path: str):
    sims = pd.read_parquet(path_to_table)

    # estimating bytes per tree node
    state = C4State()
    node1 = NodeMinimax()
    node2 = NodeMCTS(state=state)
    bytes_per_node = (sys.getsizeof(node1) + sys.getsizeof(node2)) / 2

    sims['bytes'] = sims['n_nodes'] * bytes_per_node

    # filter for sims where depth and budget align with the best depths
    filtered_sims = sims[sims.apply(lambda row: row['depth'] == best_depths.get(row['budget_total']), axis=1)]
    
    # Map agents to numeric values (case should be consistent here)
    filtered_sims['agent_numeric'] = filtered_sims['agent_curr'].map({'minimax': 0, 'mcts': 1})

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))  

    # top row (time per move)
    for i, budget in enumerate(best_depths.keys()):
        ax = axes[0, i] 
        budget_sims = filtered_sims[filtered_sims['budget_total'] == budget]

        sns.boxplot(x='agent_curr', y='ms', data=budget_sims, ax=ax, showfliers=False, width=0.3,
                    boxprops=dict(facecolor='none'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=5), zorder=10)
        
        # optional but useful for clarity
        budget_sims = budget_sims.sort_values(by='move_id', ascending=False)

        sc = ax.scatter(budget_sims['agent_numeric'], budget_sims['ms'], 
                        c=budget_sims['move_id'], cmap='viridis', edgecolor='none', alpha=1, zorder=5)

        ax.set_title(f'Budget: {budget}', fontsize=16)
        ax.set_xlabel('Agent', fontsize=14)
        ax.set_ylabel('Time (ms)', fontsize=14)
        ax.grid(alpha=0.8)

        fig.colorbar(sc, ax=ax, label='Move ID')

    # bottom row (memory usage)
    for i, budget in enumerate(best_depths.keys()):
        ax = axes[1, i]

        budget_sims = filtered_sims[filtered_sims['budget_total'] == budget]
        
        sns.boxplot(x='agent_curr', y='bytes', data=budget_sims, ax=ax, showfliers=False, width=0.3,
                    boxprops=dict(facecolor='none'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=5), zorder=10)

        budget_sims = budget_sims.sort_values(by='move_id', ascending=False)

        sc = ax.scatter(budget_sims['agent_numeric'], budget_sims['bytes'], 
                        c=budget_sims['move_id'], cmap='viridis', edgecolor='none', alpha=1, zorder=5)

        ax.set_xlabel('Agent', fontsize=14)
        ax.set_ylabel('Memory Usage (Bytes)', fontsize=14)
        ax.grid(alpha=0.8)

        fig.colorbar(sc, ax=ax, label='Move ID')

    fig.text(0.02, 0.95, 'a', ha='center', fontsize=18, weight='bold')
    fig.text(0.02, 0.48, 'b', ha='center', fontsize=18, weight='bold')

    plt.subplots_adjust(hspace=0.3)  
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

def fig_3(path_to_table: str, best_depths: dict, out_path: str):
    sims = pd.read_parquet(path_to_table)

    # filter for sims where depth and budget align with the best depths
    filtered_sims = sims[sims.apply(lambda row: row['depth'] == best_depths.get(row['budget_total']), axis=1)]

    # include only moves that resulted in a win
    winning_moves = filtered_sims[filtered_sims['is_win']]
    mean_moves_to_win = winning_moves.groupby(['agent_curr', 'mcts_strategy', 'budget_total'])['move_id'].mean().reset_index()

    # starting agent == winning agent?
    winning_moves['is_starting_agent_winner'] = (winning_moves['agent_curr'] == winning_moves['agent_start'])
    winning_moves['outcome'] = winning_moves['is_starting_agent_winner'].map({True: 'win', False: 'loss'})

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)

    # heatmaps (top-row)
    for idx, budget in enumerate(best_depths.keys()):
        budget_moves = winning_moves[winning_moves['budget_total'] == budget]
        contingency_table = pd.crosstab(budget_moves['agent_start'], budget_moves['outcome'])
        sns.heatmap(contingency_table, annot=True, cmap="viridis", fmt="d", cbar=False, ax=axes[0, idx])
        axes[0, idx].set_title(f'Budget: {budget}', fontsize=14)
        axes[0, idx].set_ylabel("Starting Agent", fontsize=12)
        axes[0, idx].set_xlabel('')  

    # point plots (bottom-row)
    for idx, budget in enumerate(best_depths.keys()):
        budget_moves = mean_moves_to_win[mean_moves_to_win['budget_total'] == budget]
        sns.pointplot(data=budget_moves, x='agent_curr', y='move_id', hue='mcts_strategy', 
                    markers='o', linestyles='-', dodge=True, ci='sd', ax=axes[1, idx], palette='Set1', legend=True if idx == 0 else False)
        axes[1, idx].set_xlabel('')
        axes[1, idx].set_ylabel('Average Moves to Win', fontsize=12)

    fig.text(0.01, 1, 'a', ha='center', fontsize=18, weight='bold')
    fig.text(0.01, 0.5, 'b', ha='center', fontsize=18, weight='bold')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

def fig_4(path_to_table_1: str, path_to_table_2: str, best_depths: dict, out_path: str):

    sims_x = pd.read_parquet(path_to_table_2)   # sims of connect 3, 5 and 6
    sims_4 = pd.read_parquet(path_to_table_1)   # sims of connect 4

    # prepare the data

    sims_4["connect"] = 4   # attributes not included in first round of simulations
    sims_4["bf"] = 7        # should be ok now

    sims_x['adjusted_win'] = (
        ((sims_x['agent_curr'] == 'minimax') & sims_x['is_win']) |
        ((sims_x['agent_curr'] == 'mcts') & sims_x['budget_exceeded'])
    )

    sims_4['adjusted_win'] = (
        ((sims_4['agent_curr'] == 'minimax') & sims_4['is_win']) |
        ((sims_4['agent_curr'] == 'mcts') & sims_4['budget_exceeded'])
    )

    sims_4_filtered = sims_4[sims_4.apply(lambda row: row['depth'] == best_depths.get(row['budget_total']), axis=1)]

    winning_sims_x = sims_x[sims_x['is_win']]
    winning_sims_3 = winning_sims_x[winning_sims_x['connect'] == 3]
    winning_sims_6 = winning_sims_x[winning_sims_x['connect'] == 6]
    winning_sims_7 = winning_sims_x[winning_sims_x['connect'] == 7]
    winning_sims_4 = sims_4_filtered[sims_4_filtered['is_win']]

    combined_winning_sims = pd.concat([winning_sims_3, winning_sims_4, winning_sims_6, winning_sims_7])

    average_moves_by_connect_budget = (combined_winning_sims.groupby(['connect', 'agent_curr', 'budget_total'])['move_id'].mean()
                                       .reset_index().rename(columns={'move_id': 'average_moves'}))

    simulations_by_connect_budget = (combined_winning_sims.groupby(['connect', 'agent_curr', 'budget_total'])['move_id'].count()
                                       .reset_index().rename(columns={'move_id': 'n_simulations'}))

    # average and n_simulations together
    average_moves_by_connect_budget = pd.merge(average_moves_by_connect_budget, simulations_by_connect_budget, on=['connect', 'agent_curr', 'budget_total'])

    # prepare outcomes for c4 simulations
    outcomes_4 = sims_4_filtered.groupby('sim_id')['adjusted_win'].any().reset_index()
    metadata_4 = sims_4_filtered[['sim_id', 'depth', 'bf', 'budget_total']].drop_duplicates()
    outcomes_4 = pd.merge(outcomes_4, metadata_4, on='sim_id')

    # prepapre outcomes for cx simulations
    outcomes_x = sims_x.groupby('sim_id')['adjusted_win'].any().reset_index()
    metadata_x = sims_x[['sim_id', 'bf', 'budget_total', 'depth']].drop_duplicates()
    outcomes_x = pd.merge(outcomes_x, metadata_x, on='sim_id')

    combined_outcomes = pd.concat([outcomes_x, outcomes_4], ignore_index=True)

    # calculate win rate
    win_rate_by_bf_budget = (combined_outcomes.groupby(['bf', 'budget_total'])['adjusted_win'].mean()
                             .reset_index().rename(columns={'adjusted_win': 'win_rate'}))

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4)  # line plot will occupy top row

    # line plot (top-row)
    ax_line = fig.add_subplot(gs[0, 1:3]) 
    sns.lineplot(
        data=win_rate_by_bf_budget,
        x='bf',
        y='win_rate',
        hue='budget_total',
        marker='o',
        linewidth=2,
        palette='tab10',
        ax=ax_line
    )

    # dashed line at 0.5 win rate (when it is better than mcts)
    ax_line.axhline(y=0.5, color='black', linestyle='--', linewidth=2) 

    ax_line.set_title('Branching Factor on Minimax Performance', fontsize=16)
    ax_line.set_xlabel('Branching Factor / C4 Columns', fontsize=14)
    ax_line.set_ylabel('Win Rate', fontsize=14)
    ax_line.grid(alpha=0.3)
    bf_values = sorted(win_rate_by_bf_budget['bf'].unique())
    ax_line.set_xticks(bf_values)
    ax_line.set_xticklabels(bf_values, fontsize=12)
    ax_line.legend(title='Budget', fontsize=12)

    # bar plots (bottom-row)
    for i, budget in enumerate(best_depths.keys()):
        ax = fig.add_subplot(gs[1, i])  

        budget_data = average_moves_by_connect_budget[average_moves_by_connect_budget['budget_total'] == budget]

        for j, connect_value in enumerate([3, 4, 6, 7]):
            connect_data = budget_data[budget_data['connect'] == connect_value]
            for agent in ['minimax', 'mcts']:
                agent_data = connect_data[connect_data['agent_curr'] == agent]

                # one case were agent has no wins!
                if agent_data.empty:
                    continue

                ax.bar(x=j + (0.2 if agent == 'mcts' else -0.2), height=agent_data['average_moves'].values[0], 
                       width=0.4, color=plt.cm.viridis(agent_data['n_simulations'].values[0] / max(budget_data['n_simulations'])), 
                       hatch={'minimax': '', 'mcts': '\\'}[agent], label=agent if j == 0 else "")

        # colorbar
        norm = plt.Normalize(vmin=min(budget_data['n_simulations']), vmax=max(budget_data['n_simulations']))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('No Simulations', fontsize=12)

        ax.set_title(f'Budget {budget}', fontsize=14)
        ax.set_xlabel('Connect Value', fontsize=12)
        ax.set_ylabel('Average No Moves', fontsize=12)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['3', '4', '6', '7'])
        ax.grid(True)
        if i == 0:
            ax.legend(title="Agent", fontsize=10)

    fig.text(0.25, 1, 'a', ha='center', fontsize=18, weight='bold')
    fig.text(0.01, 0.5, 'b', ha='center', fontsize=18, weight='bold')    

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
