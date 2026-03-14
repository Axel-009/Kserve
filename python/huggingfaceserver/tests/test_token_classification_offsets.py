import pytest
import torch

from huggingfaceserver.encoder_model import HuggingfaceEncoderModel


@pytest.fixture(scope="module")
def bert_token_classification_return_offsets_mapping():
    model = HuggingfaceEncoderModel(
        "bert-large-cased-finetuned-conll03-english",
        model_id_or_path="dbmdz/bert-large-cased-finetuned-conll03-english",
        do_lower_case=True,
        add_special_tokens=False,
        dtype=torch.float32,
        return_offsets_mapping=True,
    )
    model.load()
    yield model
    model.stop()


@pytest.mark.asyncio
async def test_bert_token_classification_return_offsets_mapping(
    bert_token_classification_return_offsets_mapping,
):
    request = "HuggingFace is a company based in Paris and New York"
    response, _ = await bert_token_classification_return_offsets_mapping(
        {"instances": [request, request]}, headers={}
    )

    assert "predictions" in response
    assert len(response["predictions"]) == 2

    assert "outputs" in response
    offset_out = next(
        (o for o in response["outputs"] if o.get("name") == "offset_mapping"),
        None,
    )
    assert offset_out is not None

    assert offset_out["datatype"] in ("INT64", "INT32")
    assert offset_out["shape"][0] == 2
    assert offset_out["shape"][-1] == 2

    seq_len_pred = len(response["predictions"][0][0])
    seq_len_offs = offset_out["shape"][1]
    assert seq_len_offs == seq_len_pred

    data = offset_out["data"]
    assert len(data) == 2 * seq_len_offs * 2
    assert all(isinstance(x, int) for x in data)
