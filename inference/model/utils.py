from collections import namedtuple

ModelOutput = namedtuple(
    "ModelOutput",
    [
        "disease_logits",
        "lesion_logits",
        "lesion_tokens",
        "cams",
        "ordinal_logits",
    ],
    defaults=(None,),
)
