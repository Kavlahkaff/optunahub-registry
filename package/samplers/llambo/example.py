import os

import optuna
from optuna import Trial
import optunahub
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def objective_rf(trial: Trial) -> float:
    """Machine learning objective using RandomForestClassifier.
    Args:
        trial: The trial object to suggest hyperparameters.
    Returns:
        Mean accuracy obtained using cross-validation.
    """
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Hyperparameter suggestions
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.01, step=0.001)

    # Define a pipeline with scaling and RandomForestClassifier
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    ccp_alpha=ccp_alpha,  # Added float parameter
                    random_state=42,
                ),
            ),
        ]
    )
    # Cross-validation for accuracy
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


if __name__ == "__main__":
    # Load the LLAMBO sampler module
    module = optunahub.load_module("samplers/llambo")

    LLAMBOSampler = module.LLAMBOSampler

    # Configuration
    sm_mode = "generative"
    max_requests_per_minute = 60
    n_trials = 30
    n_jobs = 1

    # Check if we're using Azure OpenAI
    use_azure = False

    # Verify required Azure environment variables
    required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_VERSION"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if not missing_vars and use_azure:
        print("Using Azure OpenAI with the following configuration:")
        api_key = os.environ["OPENAI_API_KEY"]
        api_base = os.environ["OPENAI_API_BASE"]
        api_version = os.environ["OPENAI_API_VERSION"]
        deployment_name = "gpt-4o"  # This is your engine name

        print(f"API Base: {api_base}")
        print(f"API Version: {api_version}")
        print(f"Deployment Name: {deployment_name}")
        print(f"API Key: {api_key[:5]}... (truncated for security)")

        use_azure = False
    else:
        print("Using standard OpenAI API...")
        # Fallback to standard OpenAI API key
        api_key = os.environ.get(
            "API_KEY",
            "",  # Put your key here or load it from environment variables.
        )
        model = "deepseek-chat"

    # Create the appropriate sampler based on available credentials
    if use_azure:
        llm_sampler = LLAMBOSampler(
            custom_task_description="Optimize RandomForest hyperparameters for digit classification.",
            sm_mode=sm_mode,
            max_requests_per_minute=max_requests_per_minute,
            api_key=api_key,
            model="gpt-4o",  # Model name should match your deployment model
            n_initial_samples=5,
            azure=True,
            azure_api_base=api_base,
            azure_api_version=api_version,
            azure_deployment_name=deployment_name,
        )
    else:
        # Create standard OpenAI sampler
        llm_sampler = LLAMBOSampler(
            custom_task_description="Optimize RandomForest hyperparameters for digit classification.",
            api_key=api_key,
            model=model,
            sm_mode=sm_mode,
            max_requests_per_minute=max_requests_per_minute,
        )

    # Create random sampler for comparison
    random_sampler = optuna.samplers.RandomSampler(seed=42)

    # Create studies
    llm_study = optuna.create_study(sampler=llm_sampler, direction="maximize")
    random_study = optuna.create_study(sampler=random_sampler, direction="maximize")

    # Run optimization
    print("Running LLM-based optimization...")
    llm_study.optimize(objective_rf, n_trials=n_trials, n_jobs=n_jobs)

    print("Running random optimization...")
    random_study.optimize(objective_rf, n_trials=n_trials, n_jobs=n_jobs)

    # Print results
    print("\nLLM-based sampler results:")
    print(f"Best accuracy: {llm_study.best_value:.4f}")
    print(f"Best parameters: {llm_study.best_params}")

    print("\nRandom sampler results:")
    print(f"Best accuracy: {random_study.best_value:.4f}")
    print(f"Best parameters: {random_study.best_params}")
