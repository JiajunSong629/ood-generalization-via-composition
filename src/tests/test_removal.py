from src.models.huggingface_models import HFModel
from src.models.removal_models import HeadRemovalModel


def test_removal():
    base_model = HFModel("llama2-70b")
    model = HeadRemovalModel(base_model=base_model)
    model.mask_top_ih(n_heads=10)
    print(model.model_meta)
    model.revert()


if __name__ == "__main__":
    test_removal()
