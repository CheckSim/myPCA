import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class myPCA:
    
    def __init__(self, df ,comp: int = 2):
        # Initialize the class with the dataframe and the desired number of principal components.
        self.components = comp
        self.dataframe = df
        
        # Check if the specified number of components is valid.
        if self.components > len(self.dataframe.columns):
            return print(f'Number of components ({self.components}) must be lower than the number of features ({len(self.dataframe.columns)})')
        
        # Create a PCA object with the specified number of components and compute the principal components.
        self.pca = PCA(n_components=self.components)
        col = []
        for i in range(self.components):
            col.append(f'PCA {i + 1}')
        self.scores = pd.DataFrame(self.pca.fit_transform(self.dataframe), columns=col)
        
    def cum_expl_variance(self):
        # Calculate and visualize the cumulative explained variance for each number of principal components.
        xint = range(1, len(self.pca.explained_variance_ratio_) + 1)
        plt.plot(xint, np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.xticks(xint)
        plt.grid(visible=True)
        plt.xlim(1, self.components, 1)
        plt.show()
        
        print(f'Cumulative explained variance with {self.components} components: {np.sum(self.pca.explained_variance_ratio_)*100}%')
        
    def biplot(self, target, PCx: int = 1, PCy: int = 2):
        # Create a biplot using the specified principal components as axes.
        
        if ((PCx > self.pca.n_components) or (PCy > self.pca.n_components)):
            return print('ERROR: PCx and PCy must be lower or equal to the number of PCA components')

        features = self.dataframe.columns
        ldngs = self.pca.components_

        fig, ax = plt.subplots(figsize=(14, 9))

        # Draw arrows for the original variables in the biplot.
        for i, feature in enumerate(features):
            ax.arrow(0, 0, ldngs[PCx-1, i], 
                     ldngs[PCy-1, i], 
                     head_width=0.03, 
                     head_length=0.03)
            ax.text(ldngs[PCx-1, i] * 1.15, 
                    ldngs[PCy-1, i] * 1.15, 
                    feature, fontsize=12)

        PCx = 'PCA ' + f'{PCx}'
        PCy = 'PCA ' + f'{PCy}'

        scaledx = 'scale ' + PCx
        scaledy = 'scale ' + PCy

        # Create a scatter plot of the scaled principal components.
        scatter = ax.scatter(self.scores[PCx].values / (self.scores[PCx].max() - self.scores[PCx].min()), 
                             self.scores[PCy].values / (self.scores[PCy].max() - self.scores[PCy].min()), 
                             c=target.values, 
                             cmap='Set3')
        
        ax.set_xlabel(PCx, fontsize=20)
        ax.set_ylabel(PCy, fontsize=20)
        ax.set_title('Biplot', fontsize=20)

        ax.legend(*scatter.legend_elements(),
                            loc="lower left", 
                            title=f"{target.name}")
