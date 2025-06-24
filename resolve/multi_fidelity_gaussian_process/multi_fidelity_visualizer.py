import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from resolve.utilities import plotting_utils as plotting
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

class MultiFidelityVisualizer():
    def __init__(self, mf_model, parameters, x_fixed):
        self.mf_model = mf_model
        self.parameters = parameters
        self.x_fixed = x_fixed
        self.color_dict = {"1sigma": 'green',"2sigma": 'yellow',"3sigma": 'coral', 
                            "hf": 'orangered',"mf": 'teal',"lf": 'lightseagreen',"lf-cnp": "teal",
                            "HF": 'orangered',"mF": 'teal',"LF": 'lightseagreen',"LF-CNP": 'teal',
                            "lf_std": 'darkturquoise', "mf_std": 'cadetblue',"hf_std": 'coral'}


        self.y_marg = None
        self.x_grid = None

    # Drawings of the model predictions projecting each dimension on a fixed point in space for the remaining dimensions
    def draw_model_projections(self, fig, outname='model_projection.png'):
        SPLIT = 100
        ncol=3

        #fig,ax = plt.subplots(nrow,ncol,figsize=(15, 5),constrained_layout=True)
        ax = fig.axes
        
        indices = [i for i in range(len(ax))]
        indices[0], indices[ncol-1] = indices[ncol-1], indices[0]

        for i,p in enumerate(self.parameters):   
            ## Compute mean and variance predictions
            x_plot=[self.x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])

            for f_idx, key in enumerate(self.mf_model.fidelities):
                f_mean_mf_model, f_var_mf_model = self.mf_model.model.predict(X_plot[f_idx*SPLIT:(f_idx+1)*SPLIT])
                f_std_mf_model = np.sqrt(f_var_mf_model)

                ax[indices[i]].fill_between(x_tmp.flatten(), (f_mean_mf_model - f_std_mf_model).flatten(), 
                            (f_mean_mf_model + f_std_mf_model).flatten(), color=self.color_dict[f"{key}_std"], alpha=0.1)
                ax[indices[i]].plot(x_tmp,f_mean_mf_model, '--', color=self.color_dict[key])

            ax[indices[i]].set_xlabel(p, fontsize=10)
            ax[indices[i]].set_ylabel(r'$y_{raw}$')
            ax[indices[i]].set_xlim(self.parameters[p][0], self.parameters[p][1])
            
        for i in range(len(self.parameters),len(ax)): 
            ax[i].set_axis_off()
        fig.savefig(outname,dpi=300, bbox_inches='tight')
        return fig

    # Drawings of the aquisition function
    def draw_acquisition_func(self, fig, us_acquisition, x_next=np.array([]), outname='model_acquisition.png'):
        SPLIT = 50
        ax2 = fig.axes

        for i, p in enumerate(self.parameters):
            ax2[i].set_title(f"Projected acquisition function - {p}")
            x_plot = [self.x_fixed[:] for _ in range(SPLIT)]
            x_tmp = np.linspace(self.parameters[p][0], self.parameters[p][1], SPLIT)
            for k in range(SPLIT):
                x_plot[k][i] = x_tmp[k]
            x_plot = np.atleast_2d(x_plot)
            X_plot = convert_x_list_to_array([x_plot, x_plot])
            
            acq = us_acquisition.evaluate(X_plot[SPLIT:])
            try:
                color = next(ax2[i].get_prop_cycle())["color"]
            except AttributeError:
                color = "blue"  # Fallback color if cycle is unavailable
            
            ax2[i].plot(x_tmp, acq / acq.max(), color=color)
            
            acq = us_acquisition.evaluate(X_plot[:SPLIT])
            ax2[i].plot(x_tmp, acq / acq.max(), color=color, linestyle="--")
            
            if x_next.any():
                ax2[i].axvline(x_next[0, i], color="red", label="x_next", linestyle="--")
                ax2[i].text(
                    x_next[0, i] + 0.5, 0.95,
                    f"x = {round(x_next[0, i], 1)}",
                    color="red", fontsize=8
                )
            
            ax2[i].set_xlabel(p)
            ax2[i].set_ylabel(r"$\mathcal{I}(x)$")
        fig.savefig(outname,dpi=300, bbox_inches='tight')
        return fig

    def model_validation(self, test_data, outname='validation.png'):
            nrows = len(next(iter(test_data.items())))
            ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 3 * nrows), squeeze=False)

            for f, (key, data) in enumerate(test_data.items()):
                ax = axes[f][0]

                if self.mf_model.normalizer.scaler_x != None:
                    x_test_tmp = self.mf_model.normalizer.transform_x(data[0])
                    y_test_tmp = self.mf_model.normalizer.transform_y(data[1])
                    #y_test_tmp = np.atleast_2d(y_test[f]).reshape(-1,1)

                else:
                    x_test_tmp = np.atleast_2d(data[0])
                    y_test_tmp = np.atleast_2d(data[1]).reshape(-1,1)
                
                x_test_tmp, y_test_tmp = (np.atleast_2d(x_test_tmp), np.atleast_2d(y_test_tmp))

                counter_1sigma = 0
                counter_2sigma = 0
                counter_3sigma = 0

                mfsm_model_mean = np.empty(shape=[0, 0])
                mfsm_model_var = np.empty(shape=[0, 0])
                y_data=[]
                x=[]
                for i in range(len(x_test_tmp)):

                        SPLIT = 1
                        x_plot = []
                        for j in range(self.mf_model.nfidelities):
                            x_plot.append((np.atleast_2d(x_test_tmp[i])))
                        X_plot = convert_x_list_to_array(x_plot)

                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[f*SPLIT:(f+1)*SPLIT])

                        y_data.append(y_test_tmp[i][0])
                        x.append(i)
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_var=np.append(mfsm_model_var,var_mf_model[0,0])
                
                if self.mf_model.normalizer.scaler_x != None:
                    y_data[:] = self.mf_model.normalizer.inverse_transform_y(y_data).flatten().tolist()
                    mfsm_model_mean = self.mf_model.normalizer.inverse_transform_y(mfsm_model_mean).flatten().tolist()
                    mfsm_model_var = self.mf_model.normalizer.inverse_transform_variance(mfsm_model_var)
                mfsm_model_std = np.sqrt(mfsm_model_var)
                
                for i in range(len(x_test_tmp)):
                    if (y_data[i] < mfsm_model_mean[i]+ mfsm_model_std[i]) and (y_data[i] > mfsm_model_mean[i]- mfsm_model_std[i]):
                                counter_1sigma += 1
                    if (y_data[i] < mfsm_model_mean[i]+2* mfsm_model_std[i]) and (y_data[i] > mfsm_model_mean[i]-2* mfsm_model_std[i]):
                                counter_2sigma += 1
                    if (y_data[i] < mfsm_model_mean[i]+3* mfsm_model_std[i]) and (y_data[i] > mfsm_model_mean[i]-3* mfsm_model_std[i]):
                                counter_3sigma += 1

                #plt.bar(x=np.arange(len(mfsm_model_mean)), height=mfsm_model_mean, color="lightgray", label='Model')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-3*mfsm_model_std, y2=mfsm_model_mean+3*mfsm_model_std, color=self.color_dict["3sigma"],alpha=0.2, label=r'$\pm 3\sigma$')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-2*mfsm_model_std, y2=mfsm_model_mean+2*mfsm_model_std, color=self.color_dict["2sigma"],alpha=0.2, label=r'$\pm 2\sigma$')
                ax.fill_between(x=np.arange(len(mfsm_model_mean)), y1=mfsm_model_mean-mfsm_model_std, y2=mfsm_model_mean+mfsm_model_std, color=self.color_dict["1sigma"],alpha=0.2, label=r'Model $\pm 1\sigma$')
                
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin,ymax*1.05)
                ax.set_ylabel(f'$\epsilon_{({key})}$')
                
                ax.plot(x[:],y_data[:],'.',color="black", label="Validation Data")
                mse = mean_squared_error(y_data, mfsm_model_mean)
                text = f"MSE: {mse:.8f} $\pm1\sigma$: {counter_1sigma/len(y_data)*100.:.0f}%  $\pm3\sigma$: {counter_2sigma/len(y_data)*100.:.0f}%  $\pm3\sigma$: {counter_3sigma/len(y_data)*100.:.0f}%"
                plotting.place_text_corner(ax, text, fontsize=11, bbox=dict(edgecolor='gray', facecolor='none', linewidth=0.5))
            
            ax.set_xlabel('Simulation Trial Number')
            legend_elements = [
                Line2D([0], [0], marker='.', color='black', linestyle='None', label='Data'),
                Line2D([0], [0], marker='.', color='white', linestyle='None', label='Model prediction'),
                mpatches.Patch(color=self.color_dict["1sigma"], alpha=0.2, label=r'$\pm 1\sigma$'),
                mpatches.Patch(color=self.color_dict["2sigma"], alpha=0.2, label=r'$\pm 2\sigma$'),
                mpatches.Patch(color=self.color_dict["3sigma"], alpha=0.2, label=r'$\pm 3\sigma$')
            ]

        
            fig.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))
            plt.tight_layout()
            fig.savefig(outname,dpi=300, bbox_inches='tight')
            return fig, [counter_1sigma/len(y_data)*100.,counter_2sigma/len(y_data)*100.,counter_3sigma/len(y_data)*100.,mse]

    def get_marginalized_approx(self, grid_steps=100, marginal_steps=500, random_seed=42):
        """
        Computes marginalized predictions for each feature separately by averaging predictions
        over all other dimensions.
        """
        if self.y_marg == None:
            np.random.seed(random_seed)
            input_dim = len(self.parameters)
            xmins = [v[0] for v in self.parameters.values()]
            xmaxs = [v[1] for v in self.parameters.values()]  
            if self.mf_model.normalizer.scaler_x != None:
                xmins = self.mf_model.normalizer.transform_x(xmins).flatten().tolist()
                xmaxs = self.mf_model.normalizer.transform_x(xmaxs).flatten().tolist()

            self.x_grid = []
            self.y_marg = {}

            for f_ix,f in enumerate(self.mf_model.fidelities):
                
                marginalized_outputs = []

                for dim, p in enumerate(self.parameters):
                    # Grid for dimension of interest
                    xmin = self.parameters[p][0]
                    xmax = self.parameters[p][1]
                    x_linspace = np.linspace(xmin, xmax, grid_steps)
                    preds_along_grid = []

                    for val in x_linspace:
                        # Random samples for other dimensions
                        X_query = np.random.uniform(xmins[dim], xmaxs[dim], size=(marginal_steps, input_dim))
                        X_query[:, dim] = val  # Fix the feature

                        y_pred = []
                        y_var = []

                        for i in range(len(X_query)):
                            SPLIT = 1
                            x_plot = []
                            for j in range(self.mf_model.nfidelities):
                                x_plot.append((np.atleast_2d(X_query[i])))
                            X_plot = convert_x_list_to_array(x_plot)


                            mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[f_ix*SPLIT:(f_ix+1)*SPLIT])
                            y_pred=np.append(y_pred,mean_mf_model[0,0])
                            y_var=np.append(y_var,var_mf_model[0, 0])
                        
                        if self.mf_model.normalizer.scaler_x != None:
                            y_pred = self.mf_model.normalizer.inverse_transform_y(y_pred)
                            y_var = self.mf_model.normalizer.inverse_transform_variance(y_var)

                        #y_pred, y_var = preds[f]
                        std_pred = np.sqrt(y_var)

                        # Sample from the GP predictive distribution
                        y_sampled = y_pred.flatten() + np.random.randn(marginal_steps) * std_pred.flatten()

                        preds_along_grid.append(y_sampled)


                    marginalized_outputs.append(np.array(preds_along_grid))  # (grid_steps, marginal_steps)

                self.x_grid = [np.linspace(self.parameters[p][0], self.parameters[p][1], grid_steps) for p in self.parameters]
                self.y_marg[f] = marginalized_outputs

        return self.y_marg, self.x_grid
    
    def get_marginalized_single_draw(self, x_data, y_data, keep_axis, x_min, x_max, scaling=1., grid_steps=25):
        """
        Marginalizes predictions over all but one feature using random sampling when only one 
        prediction is available per sample (y_hf has shape (n_samples, 1)).
        """

        x_keep = x_data[:, keep_axis]

        # Define bins along the kept axis in the original scale.
        bin_edges = np.linspace(x_min[keep_axis], x_max[keep_axis], grid_steps + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # For each bin, compute the median and 1σ percentiles (16th and 84th) from the samples in the bin.
        medians = np.empty(grid_steps)
        lower_vals = np.empty(grid_steps)
        upper_vals = np.empty(grid_steps)
        for i in range(grid_steps):
            # Use a half-open interval except for the last bin.
            if i < grid_steps - 1:
                mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_values = y_data[mask]
                medians[i] = np.median(bin_values)
                lower_vals[i] = np.percentile(bin_values, 16)
                upper_vals[i] = np.percentile(bin_values, 84)
            else:
                medians[i] = np.nan
                lower_vals[i] = np.nan
                upper_vals[i] = np.nan

        # Compute errors for plotting (errorbars represent the distance from the median to the percentiles)
        lower_error = medians - lower_vals
        upper_error = upper_vals - medians
        
        return bin_centers, medians, lower_error, upper_error
    
    def plot_marginalized(self, x_test, y_test, grid_steps=100, option = "interpol1d", smoothing=3, outname='marginalized.png'):

        xmins = [v[0] for v in self.parameters.values()]
        xmaxs = [v[1] for v in self.parameters.values()]  
        if self.y_marg == None:
            self.y_marg, self.x_grid = self.get_marginalized_approx(grid_steps=grid_steps, marginal_steps=500)
        n_fidelities = len(self.y_marg)
        n_features = len(self.x_grid)
        fig, axes = plt.subplots(nrows=n_fidelities, ncols=n_features, figsize=(5 * n_features, 4 * n_fidelities), sharex='col', sharey='row', squeeze=False)

        for f_idx, (key, marginals) in enumerate(self.y_marg.items()):
            for dim,p in enumerate(self.parameters):
                ax = axes[f_idx][dim]

                y_samples = marginals[dim]  # Shape: (grid_steps, marginal_steps)
                x_vals = self.x_grid[dim]

                y_mean = np.percentile(y_samples, 50., axis=1)
                y_1sigma_low = np.percentile(y_samples, 16., axis=1)
                y_1sigma_high = np.percentile(y_samples, 84., axis=1)
                y_2sigma_low = np.percentile(y_samples, 2.5, axis=1)
                y_2sigma_high = np.percentile(y_samples, 97.5, axis=1)
                y_3sigma_low = np.percentile(y_samples, 0.5, axis=1)
                y_3sigma_high = np.percentile(y_samples, 99.5, axis=1)

                # Fine grid for smooth plotting

                x_fine = np.linspace(x_vals[0], x_vals[-1], grid_steps)

                def smooth(y,s,option="interp1d"):
                    if option=="moving_avg":
                        return np.convolve(y, np.ones(s)/s, mode='same') #s=5
                    elif option == "spline":
                        spline = UnivariateSpline(x_vals, y, s=s)  # s=0.001 controls smoothness
                        return spline(x_fine)
                    else:
                        y_inter = interp1d(x_vals, y, kind="cubic", fill_value="extrapolate")(x_fine)
                        return gaussian_filter1d(y_inter, sigma=s) #s=2
                    
                ax.fill_between(x_fine, smooth(y_3sigma_low,smoothing,option), smooth(y_3sigma_high,smoothing,option), color=self.color_dict["3sigma"], alpha=0.2, label=r'$\pm 3\sigma$')
                ax.fill_between(x_fine, smooth(y_2sigma_low,smoothing,option), smooth(y_2sigma_high,smoothing,option), color=self.color_dict["2sigma"], alpha=0.3, label=r'$\pm 2\sigma$')
                ax.fill_between(x_fine, smooth(y_1sigma_low,smoothing,option), smooth(y_1sigma_high,smoothing,option), color=self.color_dict["1sigma"], alpha=0.3, label=r'$\pm 1\sigma$')

                ax.plot(x_vals, smooth(y_mean,smoothing,option), color="black", lw=0.8, label="Model")
                if x_test != None:
                    x, y, _, _ = self.get_marginalized_single_draw(np.array(x_test[f_idx]),np.array(y_test[f_idx]),dim,xmins,xmaxs,grid_steps=500)
                    ax.scatter(x,y,marker='.', color="black", label="Data")

                ax.set_xlabel(p, fontsize=16)
                ax.set_ylabel(f"Marginalized predicted $\\epsilon^{({key})}$", fontsize=16)

                ax.tick_params(axis='both', which='major', labelsize=10)

        n_rows = axes.shape[0]
        for i in range(n_rows - 1):
            for ax in axes[i, :]:
                ax.set_xlabel("")
        
        for i in range(n_rows):
            for ax in axes[i, 1:]:
                ax.set_ylabel("")

        # Create shared legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Model prediction'),
            mpatches.Patch(color=self.color_dict["1sigma"], alpha=0.3, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color=self.color_dict["2sigma"], alpha=0.3, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color=self.color_dict["3sigma"], alpha=0.2, label=r'$\pm 3\sigma$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=7, label='Data Points')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize='medium', frameon=False, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(outname,dpi=300, bbox_inches='tight')

        return fig

    def plot_data_marginalized(self, data, grid_steps=100, outname='marginalized_data.png'):
        markersize = 10

        xmins = [v[0] for v in self.parameters.values()]
        xmaxs = [v[1] for v in self.parameters.values()]  

        n_fidelities = len(data)
        n_features = len(xmins)
        legend_elements = []
        fig, axes = plt.subplots(nrows=n_fidelities, ncols=n_features, figsize=(5 * n_features, 4 * n_fidelities), squeeze=True, sharex='col', sharey='row')

        for f_idx, (key, data_f) in enumerate(data.items()):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_dict[key], markersize=markersize, label=f'Data $\\epsilon^{({key})}$'))
            for dim,p in enumerate(self.parameters):
                x, y, _, _ = self.get_marginalized_single_draw(np.array(data_f[0]),np.array(data_f[1]),dim,xmins,xmaxs,grid_steps=grid_steps)
                ax = axes[f_idx][dim]
                ax.scatter(x,y,marker='.', color=self.color_dict[key], s=markersize*5,label=f"Data ({key})")

                ax.set_xlabel(p, fontsize=16)
                ax.set_ylabel(f"Marginalized $\\epsilon$", fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=10)
                     

        # Create shared legend

        n_rows = axes.shape[0]
        for i in range(n_rows - 1):
            for ax in axes[i, :]:
                ax.set_xlabel("")
        
        for i in range(n_rows):
            for ax in axes[i, 1:]:
                ax.set_ylabel("")


        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize=16, frameon=False, bbox_to_anchor=(0.75, 1.05))
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(outname,dpi=300, bbox_inches='tight')

        return fig

    """
    def draw_model_marginalized(self, keep_axis=0, grid_steps=10):
            x_grid_list = []
            for p in self.parameters:
                
                arr = np.linspace(self.mf_model.normalizer.inverse_transform_x(self.parameters[p][0]),self.mf_model.normalizer.inverse_transform_x(self.parameters[p][1]), grid_steps)
                x_grid_list.append(arr)

            mesh = np.meshgrid(*x_grid_list, indexing='ij')
            points = np.stack([m.flatten() for m in mesh], axis=1)
            mesh_grid_list = points.tolist()

            nrows = self.mf_model.nfidelities
            ncols = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 5 * nrows), squeeze=False)

            for j in range(self.mf_model.nfidelities):
                ax = axes[j][0]
                mfsm_model_mean = []
                mfsm_model_var = []
                x_train = mesh_grid_list
                if j == 0:
                     print(f"Warning: There are {self.mf_model.nfidelities}x {len(x_train)} grid points to process... This can take a while")
                #y_train = self.mf_model.trainings_data[f][1]
                x_train = np.atleast_2d(x_train)

                for i in range(len(x_train)):
                        SPLIT = 1
                        x_plot = (np.atleast_2d(x_train[i]))
                        X_plot = convert_x_list_to_array([x_plot , x_plot, x_plot])
                        mean_mf_model, var_mf_model = self.mf_model.model.predict(X_plot[j*SPLIT:(j+1)*SPLIT])
                        mfsm_model_mean=np.append(mfsm_model_mean,mean_mf_model[0,0])
                        mfsm_model_var.append(var_mf_model[0, 0])

                #mfsm_model_mean = self.mf_model.normalizer.inverse_transform_y(mfsm_model_mean).flatten().tolist()
                #mfsm_model_var = self.mf_model.normalizer.inverse_transform_variance(mfsm_model_var)
                y_mean = np.array(mfsm_model_mean)
                y_var = np.array(mfsm_model_var)

                grid_shape = [grid_steps] * len(self.parameters)
                y_mean_grid = y_mean.reshape(grid_shape)
                y_var_grid = y_var.reshape(grid_shape)
                all_grid_axes = list(range(len(self.parameters)))
                marginalize_axes = tuple(ax for ax in all_grid_axes if ax != keep_axis)
                # Compute the marginal mean by averaging over the inactive dimensions.
                y_marginalized = np.mean(y_mean_grid, axis=marginalize_axes)
                # Compute the marginal variance via the law of total variance:
                #   var_marg = mean(var_grid) + variance(y_grid)
                y_var_marginalized = np.mean(y_var_grid, axis=marginalize_axes) + np.var(y_mean_grid, axis=marginalize_axes)
                y_std_marginalized = np.sqrt(y_var_marginalized)

                x_grid_transformed = x_grid_list[keep_axis]
                ax.fill_between(x_grid_transformed, y1=y_marginalized-3*y_std_marginalized, y2=y_marginalized+3*y_std_marginalized, color=self.color_dict["3sigma"],alpha=0.2, label=f'{j} model mean $\pm 3\sigma$')
                ax.fill_between(x_grid_transformed, y1=y_marginalized-2*y_std_marginalized, y2=y_marginalized+2*y_std_marginalized, color=self.color_dict["2sigma"],alpha=0.2, label=f'{j} model mean $\pm 2\sigma$')
                ax.fill_between(x_grid_transformed, y1=y_marginalized-y_std_marginalized, y2=y_marginalized+y_std_marginalized, color=self.color_dict["1sigma"],alpha=0.2, label=f'{j} model mean $\pm 1\sigma$')
                ax.plot(x_grid_transformed,y_marginalized,color=self.colors_mean[j], label=f'{j} model mean')
                #ax.plot(x[:],y_data[:],'.',color="black", label="Data")
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin*0.95,ymax*1.05)
                #handles, labels = ax[j].get_legend_handles_labels()
                
                #order = [2,1,0]
                #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9, bbox_to_anchor=(0.665,1.),ncol=3)
                ax.set_ylabel('Data and Model Prediction')
                if j == (self.mf_model.nfidelities-1):
                    ax.set_xlabel(list(self.parameters.keys())[keep_axis])
            
            return fig
        """
    