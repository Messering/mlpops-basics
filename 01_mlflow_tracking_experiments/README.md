# MLflow Demo Project - Educational Materials

This project demonstrates MLflow capabilities for learning ML experiment tracking

## 🚀 Quick Start

### 1. Installation

```bash
pip install mlflow scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run Notebooks in Order

Open and run notebooks sequentially:
1. **01_basic_tracking.ipynb** - Learn basic MLflow tracking
2. **02_hyperparameter_tuning.ipynb** - Experiment with hyperparameters
3. **03_model_comparison.ipynb** - Compare different ML algorithms
4. **04_advanced_tracking.ipynb** - Advanced MLflow features
5. **05_model_registry.ipynb** - Model versioning and deployment

### 3. View Results in MLflow UI

After running any notebook, start the MLflow UI:

```bash
cd mlflow_demo_project
mlflow ui
```

Then open: http://localhost:5000

## 📖 Learning Objectives

### Notebook 1: Basic Tracking
- Set up MLflow experiments
- Log parameters and metrics
- Save models as artifacts
- View results in MLflow UI

### Notebook 2: Hyperparameter Tuning
- Run multiple experiments with different parameters
- Compare model performance
- Identify best configurations
- Visualize parameter impact

### Notebook 3: Model Comparison
- Compare different ML algorithms
- Track multiple metrics
- Generate comparison visualizations
- Select best model architecture

### Notebook 4: Advanced Features
- Use tags for organization
- Log complex artifacts (plots, datasets)
- Implement nested runs
- Use autologging
- Custom metrics and parameters

### Notebook 5: Model Registry
- Register models
- Manage model versions
- Promote models through stages (Staging → Production)
- Deploy models from registry

## 🎯 MLflow UI Features to Explore

1. **Experiments View** - Compare runs side-by-side
2. **Metrics Visualization** - Plot metrics across runs
3. **Parameter Comparison** - Analyze parameter impact
4. **Artifact Browser** - View saved models and plots
5. **Model Registry** - Manage production models

## 💡 Tips

- Run cells sequentially to understand the flow
- Experiment with different parameters
- Check MLflow UI after each notebook
- Compare your results with classmates
- Try modifying the code and observe changes

## �🛠️ Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'seaborn'`
**Solution**: `pip install seaborn`

**Issue**: MLflow UI shows "landing page not found"
**Solution**: 
```bash
pip uninstall mlflow
pip install mlflow
```

**Issue**: Kernel needs restart after installing packages
**Solution**: Restart kernel in Jupyter and re-run cells

## 📊 Expected Outcomes

After completing all notebooks, your MLflow UI will show:
- 10+ experiments with multiple runs each
- Various model types and configurations
- Comparative visualizations
- Registered models in different stages
- Complete audit trail of all experiments

## 📝 Assignment Ideas

1. Run all notebooks and screenshot your MLflow UI
2. Create a new experiment with your own dataset
3. Find the best model configuration for a specific metric
4. Register and promote a model to Production stage
5. Export a model and create predictions
