from importlib import import_module

import typer

app = typer.Typer()


def run_main(module_name: str):
    module = import_module(f"{module_name}")
    module.main()


@app.command()
def load():
    """Run the load.py script."""
    run_main("ml.load")


@app.command()
def train_expected_returns():
    """Run the train_expected_returns.py script."""
    run_main("ml.train_expected_returns")


@app.command()
def predict_expected_returns():
    """Run the predict_expected_returns.py script."""
    run_main("ml.predict_expected_returns")


@app.command()
def predict_covariance_matrix():
    """Run the predict_covariance_matrix.py script."""
    run_main("ml.predict_covariance_matrix")


@app.command()
def optimize():
    run_main("optimization.mvo")


if __name__ == "__main__":
    app()
