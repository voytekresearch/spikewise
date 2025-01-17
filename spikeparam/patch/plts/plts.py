"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np



def plot_model(model, inds=None, mode='full', in_ms=True, show_points=False, ax=None, groups=False, ind_groups=None,  group_names=None,  plot_average=False, plot_average_std=False, color_spks='C0'):
    """Plot model results.

    Parameters
    ----------
    model : Spike
        Fit Spike object.
    inds : {None, int, list of int}
        Spike indices to plot.
    mode : {'full', 'ramp', 'exp'}
        Whether to plot the full spike, or ramping/exponential decay portions.
    in_ms : bool, optional, default: True
        Scales x-axis to ms if True.
    show_points : bool, optional, default: False
        Show control points used to segment the spike.
    ax : axis
        Axis to plot on.
    """
    # Plot
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))

    wght = 1000 if in_ms else 1

    lab_true = 'Actual'
    lab_fit = 'Fit'

    if isinstance(inds, int):
        inds = [inds]
    elif inds is None:
        inds = range(len(model.spikes))

    # Dynamic alpha
    alpha = 1/(len(inds)**.33)

    # Plot full fit
    _times = model.times * wght

    colors = ['firebrick', 'royalblue']
    if plot_average:

        # Create custom legend handles and labels
        custom_legend_handles = []

        if groups:
            if ind_groups is None:
                raise ValueError("ind_groups cannot be None when groups is True. Please provide group indeces for the overlay group plots.")

            if group_names is None:
                group_names = [f'Group {i}' for i in range(len(ind_groups))]

            if plot_average:
                for idx, group in enumerate(ind_groups):
                    if idx > len(colors) - 1:
                        color = plt.cm.viridis(float(idx) / len(ind_groups))
                    else:
                        color = colors[idx]
                    avg_actual = np.mean([model.spikes[i] for i in group], axis=0)

                    ax.plot(_times, avg_actual, color=color, alpha=0.7, label=f'Avg Actual {group_names[idx]}', linewidth=5)

                    custom_legend_handles.extend([plt.Line2D([0], [0], color=color, marker='.', markersize=8, label=f'Average Actual {group_names[idx]}')])

                    if plot_average_std:
                        #plot std of the waveform across time for groups 
                        std_dev = np.std([model.spikes[i] for i in group], axis=0)
                        # Visualize standard deviation as a shaded band
                        ax.fill_between(_times, avg_actual - std_dev, avg_actual + std_dev, color=color, alpha=0.3)
                        # Add shaded standard deviation band to the legend
                        custom_legend_handles.append(plt.fill_between([], [], [], color=color, alpha=0.3, label=f'Std Dev Band {group_names[idx]}'))




        else:

            #plot average for all data - no groups 
            avg_actual = np.mean(model.spikes, axis=0)
            ax.plot(_times, avg_actual, color=color_spks, label=f'Avg Actual', linewidth=5)
            if plot_average_std:
                #plot std of the waveform across time for groups 
                std_dev = np.std(model.spikes, axis=0)
                # Visualize standard deviation as a shaded band
                ax.fill_between(_times, avg_actual - std_dev, avg_actual + std_dev, color=color_spks, alpha=0.3)
    
            custom_legend_handles.extend([plt.Line2D([0], [0], color=color_spks, marker='.', markersize=8, label='Average Actual')])
            # Add shaded standard deviation band to the legend
            custom_legend_handles.append(plt.fill_between([], [], [], color=color_spks, alpha=0.3, label='Std Dev Band'))


    else:

        if mode == 'full':

            for i in inds:

                if i in model.inds_error:
                    continue

                ax.plot(_times, model.spikes[i], color=color_spks, label=lab_true, alpha=alpha)
                lab_true = ''

                if show_points:
                    _plot_control_points(_times, model.spikes[i], model.indices[i], ax)

            for i in inds:

                if i in model.inds_error:
                    continue

                # Ramp
                
                start, end = model.indices[i][0], model.indices[i][1]
                if len(_times[start:end]) != len(model.fit_ramp[i]):
                    continue
                    
                ax.plot(_times[start:end], model.fit_ramp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

                # Exponential
                start, end = model.indices[i][-2], model.indices[i][-1]
                ax.plot(_times[start:end], model.fit_exp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')

        # Only plot ramp fit
        elif mode == 'ramp':

            for i in inds:
                if i in model.inds_error:
                    continue

                start, end = model.indices[i][0], model.indices[i][1]
                ax.plot(_times[start:end], model.spikes[i][start:end], color=color_spks,
                        label=lab_true, alpha=alpha)
                lab_true = ''

            for i in inds:

                if i in model.inds_error:
                    continue

                ax.plot(_times[start:end], model.fit_ramp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

        # Only plot exp fit
        elif mode == 'exp':

            for i in inds:
                if i in model.inds_error:
                    continue

                start, end = model.indices[i][-2], model.indices[i][-1]
                ax.plot(_times[start:end], model.spikes[i][start:end], color=color_spks,
                        label=lab_true, alpha=alpha)
                lab_true = ''

            for i in inds:

                if i in model.inds_error:
                    continue

                ax.plot(_times[start:end], model.fit_exp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

        ax.set_ylabel('Voltage')
        ax.set_xlabel('Time (ms)')

        # Create custom legend handles and labels
        custom_legend_handles = []

       
        
        custom_legend_handles.extend([plt.Line2D([0], [0], color=color_spks, marker='.', markersize=8, label='Actual'),
                                           plt.Line2D([0], [0], color='C1', linestyle='--', label='Fit')])

        if show_points:
            custom_legend_handles.extend([plt.Line2D([0], [0], marker='o', markersize=8, label='Start', color='C2'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Inflection', color='C3'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Rise', color='C4'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Peak', color='C5'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Decay', color='C6')])

        


        if groups:
            if ind_groups is None:
                raise ValueError("ind_groups cannot be None when groups is True. Please provide group indeces for the overlay group plots.")

            if group_names is None:
                group_names = [f'Group {i}' for i in range(len(ind_groups))]

            # If plot_average is False, plot individual spikes for each group
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                for i in group:
                    if i in model.inds_error:
                        continue

                    if mode == 'full':
                        ax.plot(_times, model.spikes[i], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        if show_points:
                            _plot_control_points(_times, model.spikes[i], model.indices[i], ax)
                        start, end = model.indices[i][0], model.indices[i][1]
                        if len(_times[start:end]) == len(model.fit_ramp[i]):
                            ax.plot(_times[start:end], model.fit_ramp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')
                        start, end = model.indices[i][-2], model.indices[i][-1]
                        ax.plot(_times[start:end], model.fit_exp[i], color=color, linestyle='--')
                    elif mode == 'ramp':
                        start, end = model.indices[i][0], model.indices[i][1]
                        ax.plot(_times[start:end], model.spikes[i][start:end], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        ax.plot(_times[start:end], model.fit_ramp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')
                    elif mode == 'exp':
                        start, end = model.indices[i][-2], model.indices[i][-1]
                        ax.plot(_times[start:end], model.spikes[i][start:end], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        ax.plot(_times[start:end], model.fit_exp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')

            # Create custom legend handles and labels
            custom_legend_handles = []

            # Add handles and labels for actual data
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                custom_legend_handles.append(plt.Line2D([0], [0], color=color, label=f'Actual {group_names[idx]}'))

            # Add handles and labels for fit data
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                custom_legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='--', label=f'Fit {group_names[idx]}'))

            # Add control points if show_points is True
            if show_points:
                for idx, group in enumerate(ind_groups):
                    color = plt.cm.viridis(float(idx) / len(ind_groups))
                    custom_legend_handles.extend([
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Start {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Inflection {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Rise {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Peak {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Decay {group_names[idx]}', color=color)
                    ])

           




    

    ax.legend(handles=custom_legend_handles)
   


def _plot_control_points(times, spike, indices, ax):

    labels = ['Start', 'Inflection', 'Rise', 'Peak', 'Decay']
    colors = ['C' + str(i) for i in range(2, 7)]
    for i, l, c in zip(indices[:-2], labels, colors):
        ax.plot(times[i], spike[i], ls='', marker='.', ms=16, label=l, color=c)
