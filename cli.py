import os

import typer

import download

app = typer.Typer()


@app.command()
def get_data():
    data = download.pull_prices()
    data.to_csv("data/prices.csv", index=False)


if __name__ == "__main__":
    app()
