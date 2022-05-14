import sys
import torch
import mlflow

from torchviz import make_dot

sys.path.append(".")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 1, 320, 320).to(device)
    logged_model = 'runs:/54db63cc351242399e8fc208d55e0ed7/model'
    model = mlflow.pytorch.load_model(logged_model)
    graph = make_dot(
        model(x),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    graph.view(cleanup=True)


if __name__ == "__main__":
    main()
