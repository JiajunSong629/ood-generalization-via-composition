from src.tasks.copying.task import CopyingTask
from src.tasks.icl.task import ICLTask
from src.models.huggingface_models import HFModel
from src.models.removal_models import HeadRemovalModel


def test_removal():
    base_model = HFModel("llama2-7b")
    removal_model = HeadRemovalModel(base_model)
    # task = CopyingTask(seg_len=10, rep=3, ignore_burning=4, ignore_segment=1)
    task = ICLTask(setting="symbol", num_shots=10)

    induction_heads = base_model.induction_heads
    L1, L2 = [i[0] for i in induction_heads][:2]
    print(L1, L2)

    print("######################## Base model")
    result = task.evaluate_model(removal_model, num_samples=10, task_random_seed=42)
    print(result)
    attn1 = base_model._model.model.layers[L1].self_attn
    attn2 = base_model._model.model.layers[L2].self_attn
    print(attn1.forward)
    print(attn2.forward)

    print("######################## Mask top ih 10")
    removal_model.mask_top_ih(10)
    result = task.evaluate_model(removal_model, num_samples=10, task_random_seed=42)
    print(result)
    print(induction_heads[:10])

    attn1 = removal_model._model.model.layers[L1].self_attn
    attn2 = removal_model._model.model.layers[L2].self_attn
    print(attn1.forward)
    print(attn2.forward)

    print("######################## Revert model")
    removal_model.revert()
    result = task.evaluate_model(removal_model, num_samples=10, task_random_seed=42)
    print(result)
    attn1 = removal_model._model.model.layers[L1].self_attn
    attn2 = removal_model._model.model.layers[L2].self_attn
    print(attn1.forward)
    print(attn2.forward)

    print("######################## Mask top ih 20")
    removal_model.mask_top_ih(20)
    result = task.evaluate_model(removal_model, num_samples=10, task_random_seed=42)
    print(result)
    attn1 = removal_model._model.model.layers[L1].self_attn
    attn2 = removal_model._model.model.layers[L2].self_attn
    print(attn1.forward)
    print(attn2.forward)


if __name__ == "__main__":
    test_removal()
