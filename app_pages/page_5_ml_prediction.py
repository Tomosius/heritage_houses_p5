import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.compose import TransformedTargetRegressor
from file_management import load_house_data, load_pkl_file

# Set seaborn style for plots
sns.set_style("whitegrid")


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """Evaluates regression model performance on training and test sets."""
    r2_train, mae_train, mse_train, rmse_train, msle_train = regression_evaluation(X_train, y_train, pipeline)
    r2_test, mae_test, mse_test, rmse_test, msle_test = regression_evaluation(X_test, y_test, pipeline)
    return (r2_train, mae_train, mse_train, rmse_train, msle_train), (r2_test, mae_test, mse_test, rmse_test, msle_test)


def regression_evaluation(X, y, pipeline):
    """Predicts and calculates various regression performance metrics."""
    prediction = pipeline.predict(X)
    r2 = r2_score(y, prediction)
    mae = mean_absolute_error(y, prediction)
    mse = mean_squared_error(y, prediction)
    rmse = np.sqrt(mse)
    msle = mean_squared_log_error(y, prediction)
    return r2, mae, mse, rmse, msle


def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    """Generates plots for evaluating regression model performance."""
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    r2_train, mae_train, mse_train, rmse_train, msle_train = regression_evaluation(X_train, y_train, pipeline)
    r2_test, mae_test, mse_test, rmse_test, msle_test = regression_evaluation(X_test, y_test, pipeline)

    # Train set: Actual vs Predicted
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0, 0], color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predictions")
    axes[0, 0].set_title("Train Set: Actual vs Predicted")
    train_metrics_text = (f'R2: {round(r2_train, 3)}\n'
                          f'MAE: {round(mae_train, 3)}\n'
                          f'MSE: {round(mse_train, 3)}\n'
                          f'RMSE: {round(rmse_train, 3)}\n'
                          f'MSLE: {round(msle_train, 3)}')
    axes[0, 0].text(0.05, 0.95, train_metrics_text, transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Test set: Actual vs Predicted
    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[0, 1], color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 1].set_xlabel("Actual Values")
    axes[0, 1].set_ylabel("Predictions")
    axes[0, 1].set_title("Test Set: Actual vs Predicted")
    test_metrics_text = (f'R2: {round(r2_test, 3)}\n'
                         f'MAE: {round(mae_test, 3)}\n'
                         f'MSE: {round(mse_test, 3)}\n'
                         f'RMSE: {round(rmse_test, 3)}\n'
                         f'MSLE: {round(msle_test, 3)}')
    axes[0, 1].text(0.05, 0.95, test_metrics_text, transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Train set: Residuals
    residuals_train = y_train - pred_train
    sns.scatterplot(x=pred_train, y=residuals_train, alpha=alpha_scatter, ax=axes[1, 0], color='blue')
    axes[1, 0].axhline(0, color='r', linestyle='--')
    axes[1, 0].set_xlabel("Predictions")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Train Set: Residuals")

    # Test set: Residuals
    residuals_test = y_test - pred_test
    sns.scatterplot(x=pred_test, y=residuals_test, alpha=alpha_scatter, ax=axes[1, 1], color='green')
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel("Predictions")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Test Set: Residuals")

    # Train set: Error Distribution
    sns.histplot(residuals_train, kde=True, ax=axes[1, 2], color='blue')
    axes[1, 2].set_xlabel("Residuals")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Train Set: Error Distribution")

    # Test set: Error Distribution
    sns.histplot(residuals_test, kde=True, ax=axes[0, 2], color='green')
    axes[0, 2].set_xlabel("Residuals")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Test Set: Error Distribution")

    plt.tight_layout()
    return fig


def plot_feature_importance_absolute(selected_pipeline):
    """
    Plot the absolute feature importance from a given pipeline.

    Args:
        selected_pipeline (Pipeline): The complete pipeline including feature selection and model.
    """

    def extract_model(pipeline):
        """Extract the final model from the pipeline."""
        for step_name, step in pipeline.steps:
            if isinstance(step, TransformedTargetRegressor):
                return step.regressor_
            elif hasattr(step, 'feature_importances_') or hasattr(step, 'coef_'):
                return step
        return None

    try:
        # Extract the sub-pipeline up to the model step
        feature_names = None
        for name, step in selected_pipeline.named_steps.items():
            if name == 'model':
                break
            if hasattr(step, 'get_feature_names_out'):
                if feature_names is None:
                    feature_names = step.get_feature_names_out()
                else:
                    feature_names = step.get_feature_names_out(feature_names)
            else:
                # Simulate the transformation to get feature names
                if feature_names is None:
                    feature_names = step.transform(pd.DataFrame(columns=[f'feature_{i}' for i in range(
                        step.transform(pd.DataFrame()).shape[1])])).columns.tolist()
                else:
                    feature_names = step.transform(pd.DataFrame(columns=feature_names)).columns.tolist()

        if feature_names is None:
            raise ValueError("Could not retrieve transformed feature names.")

        # Extract the final model from the pipeline
        model = extract_model(selected_pipeline)

        if model is None:
            raise ValueError("The model does not have feature importances or coefficients.")

        # Get feature importances or coefficients
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = 'importance'
        elif hasattr(model, 'coef_'):
            importances = model.coef_.flatten()
            importance_type = 'coefficient'
        else:
            raise ValueError("The model does not have feature importances or coefficients.")

        # Create a DataFrame for feature importances or coefficients
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            importance_type: importances
        }).sort_values(by=importance_type, ascending=False)

        # Plotting the feature importances or coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance_type, y='Feature', data=feature_importances_df)
        plt.xlabel(importance_type.capitalize())
        plt.ylabel('Feature')
        plt.title(f'Feature {importance_type.capitalize()}s')
        st.pyplot(plt.gcf())

    except KeyError as e:
        st.error(f"KeyError: {e}")
    except AttributeError as e:
        st.error(f"AttributeError: {e}")
    except ValueError as e:
        st.error(f"ValueError: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def display():
    """Displays the Streamlit app interface for evaluating a house price prediction model."""
    df = load_house_data()
    X = df.drop(columns='SalePrice')
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = load_pkl_file()

    st.title("House Price Prediction Model Evaluation")
    st.write(
        "This application evaluates the performance of a house price prediction model, providing insights into model "
        "accuracy and the importance of various features.")

    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(df)

    st.subheader('Model Performance Metrics')
    train_metrics, test_metrics = regression_performance(X_train, y_train, X_test, y_test, model)

    st.write("### Training Set Metrics")
    st.write(f"**R2:** {train_metrics[0]:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {train_metrics[1]:.3f}")
    st.write(f"**Mean Squared Error (MSE):** {train_metrics[2]:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {train_metrics[3]:.3f}")
    st.write(f"**Mean Squared Log Error (MSLE):** {train_metrics[4]:.3f}")

    st.write("### Test Set Metrics")
    st.write(f"**R2:** {test_metrics[0]:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {test_metrics[1]:.3f}")
    st.write(f"**Mean Squared Error (MSE):** {test_metrics[2]:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {test_metrics[3]:.3f}")
    st.write(f"**Mean Squared Log Error (MSLE):** {test_metrics[4]:.3f}")

    st.subheader('Evaluation Plots')
    st.write(
        "The following plots provide a visual representation of the model's performance on both the training and test "
        "sets.")
    fig = regression_evaluation_plots(X_train, y_train, X_test, y_test, model)
    st.pyplot(fig)

    st.subheader('Feature Importance')
    st.write("The bar plot below shows the absolute importance of features in predicting the house prices.")
    plot_feature_importance_absolute(model)
