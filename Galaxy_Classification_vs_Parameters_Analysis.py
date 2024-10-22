import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy.stats import pointbiserialr

def analyze_parameters(df, parameters, classification_col='flag', 
                      save_plots=False, output_dir='./plots/'):
    """
    Analyze relationship between galaxy classification and multiple parameters
    
    Parameters:
    df: pandas DataFrame
    parameters: list of column names to analyze
    classification_col: name of the classification column (default='classification')
    save_plots: boolean, whether to save plots to files
    output_dir: directory to save plots if save_plots=True
    
    Returns:
    dict: Summary of statistical results for each parameter
    """
    results = {}
    
    for param in parameters:
        print(f"\n{'='*50}")
        print(f"Analyzing parameter: {param}")
        print('='*50)
        
        # Store results for this parameter
        param_results = {}
        
        # Basic Statistics
        stats_by_class = df.groupby(classification_col).agg({
            param: ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        print("\nStatistics by Classification:")
        print("0 = Spirals, 1 = Ellipticals")
        print(stats_by_class)
        param_results['basic_stats'] = stats_by_class
        
        # Point-Biserial Correlation
        correlation, p_value = pointbiserialr(df[classification_col], df[param])
        print(f"\nPoint-Biserial Correlation:")
        print(f"Correlation coefficient: {correlation:.3f}")
        print(f"P-value: {p_value:.3f}")
        param_results['correlation'] = {
            'coefficient': correlation,
            'p_value': p_value
        }
        
        # Create visualizations
        fig = plt.figure(figsize=(15, 10))
        
        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(x=classification_col, y=param, data=df)
        plt.title(f'{param} Distribution by Galaxy Type')
        plt.xlabel('Galaxy Type (0=Spiral, 1=Elliptical)')
        plt.ylabel(param)
        
        # Violin plot
        plt.subplot(2, 2, 2)
        sns.violinplot(x=classification_col, y=param, data=df)
        plt.title(f'{param} Distribution Density by Galaxy Type')
        plt.xlabel('Galaxy Type (0=Spiral, 1=Elliptical)')
        plt.ylabel(param)
        
        # Histogram
        plt.subplot(2, 2, 3)
        for class_type in [0, 1]:
            sns.histplot(data=df[df[classification_col] == class_type][param], 
                        label=f'Type {class_type}',
                        alpha=0.5,
                        bins=20)
        plt.title(f'{param} Histogram by Galaxy Type')
        plt.xlabel(param)
        plt.ylabel('Count')
        plt.legend()
        
        # ROC Curve
        plt.subplot(2, 2, 4)
        X = df[param].values.reshape(-1, 1)
        y = df[classification_col]
        
        model = LogisticRegression()
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {param} as Classifier')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}{param}_analysis.png")
        
        # Mann-Whitney U test
        statistic, mw_pvalue = stats.mannwhitneyu(
            df[df[classification_col] == 0][param],
            df[df[classification_col] == 1][param],
            alternative='two-sided'
        )
        print("\nMann-Whitney U Test:")
        print(f"Statistic: {statistic:.3f}")
        print(f"P-value: {mw_pvalue:.3f}")
        param_results['mann_whitney'] = {
            'statistic': statistic,
            'p_value': mw_pvalue
        }
        
        # Calculate effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_se
        
        d = cohens_d(df[df[classification_col] == 0][param],
                     df[df[classification_col] == 1][param])
        print(f"\nCohen's d effect size: {d:.3f}")
        param_results['cohens_d'] = d
        
        # Determine statistical significance
        is_significant = (p_value < 0.05) or (mw_pvalue < 0.05)
        significance_strength = None
        
        if is_significant:
            if abs(d) < 0.2:
                significance_strength = "Statistically significant but negligible effect size"
            elif abs(d) < 0.5:
                significance_strength = "Statistically significant with small effect size"
            elif abs(d) < 0.8:
                significance_strength = "Statistically significant with medium effect size"
            else:
                significance_strength = "Statistically significant with large effect size"
        else:
            significance_strength = "Not statistically significant"
        
        print(f"\nOverall Assessment: {significance_strength}")
        param_results['significance_assessment'] = significance_strength
        
        results[param] = param_results
        plt.show()
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    for param, res in results.items():
        print(f"\n{param}:")
        print(f"Assessment: {res['significance_assessment']}")
        print(f"Correlation coefficient: {res['correlation']['coefficient']:.3f}")
        print(f"Effect size (Cohen's d): {res['cohens_d']:.3f}")
    
    return results


fuji_pv = pd.read_csv("DESI_FP_logdists_fiducial.csv")
parameters_to_test = ['z_x', 'r', 's', 'i', 'mag_r', 'ra_1', 'dec_1', 'absmag_r']
results = analyze_parameters(fuji_pv, parameters_to_test)