"""
Implement DistilBERT by Sanh et al. 2019 (arXiv 1910.01108)

DistilBERT code taken from the HuggingFace Transformer 2.11.0 library with minor modifications
Allen NLP compatibility code taken from https://github.com/allenai/allennlp/pull/4495/files
"""

from copy import deepcopy
import math
import torch
from torch import Tensor
import transformers
import numpy as np

from transformers import AutoModel, PreTrainedModel
from typing import Dict, List, Optional, Iterable, Union
from allennlp.data.batch import Batch
from allennlp.common import JsonDict
from allennlp.data import TextFieldTensors, Vocabulary, Instance
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
import torch.nn as nn


# Local imports
from model import CaptumCompatible
from modules.architectures.transformer import Transformer, Embeddings
from modules.attention.activations import SoftmaxActivation
from modules.attention.attention import Attention, AttentionAnalysisMethods
from modules.attention.self import MultiHeadSelfAttention

from util import Config


def make_vanilla_distilbert(args, meta):
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=meta.num_targets
    )
    return model


################################
# DistillBert based classifier #
################################
# Code mainly taken from court-of-xai; in turn taken from huggingface/transformers


class DistilBertEncoder(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 12,
        dim: int = 768,
        hidden_dim: int = 4 * 768,
        ffn_activation: str = "gelu",
        ffn_dropout: float = 0.2,
        attention: Attention = MultiHeadSelfAttention(
            n_heads=12, dim=768, activation_function=SoftmaxActivation(), dropout=0.2
        ),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.ffn_activation = ffn_activation
        self.ffn_dropout = ffn_dropout

        self.transformer = Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            ffn_activation=self.ffn_activation,
            ffn_dropout=self.ffn_dropout,
            attention=attention,
        )

    @classmethod
    def from_huggingface_model(
        cls,
        model: PreTrainedModel,
        # ffn_activation: str,
        # ffn_dropout: float,
        # attention: Attention
    ):
        config = model.config
        encoder = cls(
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            # ffn_activation=ffn_activation,
            # ffn_dropout=ffn_dropout,
            # attention=attention
        )
        # After creating the encoder, we copy weights over from the transformer.  This currently
        # requires that the internal structure of the text side of this encoder *exactly matches*
        # the internal structure of whatever transformer you're using.
        encoder_parameters = dict(encoder.named_parameters())
        for name, parameter in model.named_parameters():
            if name.startswith("transformer."):
                name = name.replace("LayerNorm", "layer_norm")
                if name not in encoder_parameters:
                    raise ValueError(
                        f"Couldn't find a matching parameter for {name}. Is this transformer "
                        "compatible with the joint encoder you're using?"
                    )
                encoder_parameters[name].data.copy_(parameter.data)

        return encoder

    def forward(
        self,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None,
        output_hidden_states: Optional[bool] = False,
    ):
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class DistilBertForSequenceClassification(torch.nn.Module, CaptumCompatible):
    def __init__(self, config, meta):
        super().__init__()
        self.vocab = meta.vocab

        # Positional embeddings + token embeddings
        # self.embedding_dim = config.embedding_dim
        self.embeddings = meta.embeddings

        # DistillBert
        self.encoder = meta.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.num_labels = meta.num_labels
        # Labels = labels; targets = output probs
        if self.num_labels == 2:
            self.num_targets = 1
        else:
            self.num_targets = self.num_labels

        self.seq_classif_dropout = config.seq_classif_dropout

        self.pre_classifier = nn.Linear(self.encoder.dim, self.encoder.dim)
        # self.classifier = nn.Linear(self.encoder.dim, self.num_labels)
        # Change: use BCEWithLogits and output a single value if there
        #         are two labels
        self.classifier = nn.Linear(self.encoder.dim, self.num_targets)
        self.dropout = nn.Dropout(self.seq_classif_dropout)

        self.supported_attention_analysis_methods = [
            AttentionAnalysisMethods.weight_based,
            AttentionAnalysisMethods.norm_based,
        ]

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = (
            torch.nn.CrossEntropyLoss()
            if self.num_targets > 1
            else torch.nn.BCEWithLogitsLoss()
        )

    @classmethod
    def from_huggingface_model_name(cls, config, meta):
        model_name = config.pretrained_model
        seq_classif_dropout = config.seq_classif_dropout
        num_labels = meta.num_labels
        vocab = meta.vocab
        transformer = AutoModel.from_pretrained(model_name)
        embeddings = deepcopy(transformer.embeddings)
        encoder = DistilBertEncoder.from_huggingface_model(
            model=transformer
            # ffn_activation=ffn_activation,
            # ffn_dropout=ffn_dropout,
            # attention=attention
        )

        # Just to be compatible with the config/meta model signature
        config = Config()
        config.seq_classif_dropout = seq_classif_dropout

        meta = Config()
        meta.vocab = vocab
        meta.embeddings = embeddings
        meta.encoder = encoder
        meta.num_labels = num_labels

        return cls(config, meta)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.metrics["accuracy"].get_metric(reset=reset)}

    def forward_inner(
        self, embedded_tokens, attention_mask, label, output_attentions, output_dict
    ):
        # (bs, seq_len) -> (num_hidden_layers, batch, num_heads, seq_length, seq_length)
        # print(attention_mask)
        # print(attention_mask.shape)
        head_mask = attention_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        head_mask = head_mask.expand(
            self.encoder.n_layers, -1, self.encoder.n_heads, -1, attention_mask.shape[1]
        )
        # print(head_mask.shape)

        encoder_output = self.encoder(
            inputs_embeds=embedded_tokens,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        hidden_state = encoder_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim) # CLS Token
        output_dict["hiddens"] = hidden_state  # B, T, D

        # Single hidden layer decoder
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        output_dict["logits"] = logits

        if output_attentions:
            # Tuple of n_layer dicts of tensors of shape (bs, ..., seq_length)
            # Stack to single tuple of (bs, n_layers, ...,  seq_length)
            attentions = encoder_output[1]

            for analysis_method in output_attentions:
                output_dict[analysis_method] = torch.stack(
                    attentions[analysis_method], dim=1
                )

        if self.num_targets == 1:
            class_probabilities = torch.nn.Sigmoid()(logits)
            # if label is not None:
            # print("[L]", logits.isnan().any())
            # print("[C]", class_probabilities.isnan().any())
            # print("[L]", logits)
            # print("[C]", class_probabilities)

        else:
            class_probabilities = torch.nn.Softmax(dim=-1)(logits)
        output_dict["class_probabilities"] = class_probabilities

        if label is not None:
            # Has to be not-None for captum interpret
            if self.num_targets == 1:
                # Binary case: p * L + (1-p) * (1-L)
                p = class_probabilities.squeeze()
                # print(p.shape)
                # print(label.shape)
                # print((p * label).shape)
                # print(((1.- p) * (1 - label)).shape)

                correct_class_pred = p * label + (1.0 - p) * (1 - label)
                # print("Debug probs")
                # print(p)
                # print(p * label)
                # print((1. - p) * (1 - label))
                # print(correct_class_pred)
                # print(correct_class_pred, '\n', p, '\n', label)
                # print(correct_class_pred.shape)
                return correct_class_pred
            else:
                # Multiclass case
                nr_classes = self.num_labels
                (B,) = label.shape
                # Copy labels nr_classes times ([0, 1] -> [[0,0],[1,1]])
                label2 = label.unsqueeze(-1).expand(B, nr_classes)
                # Obtain prediction probabilities for _correct class_
                mask = (
                    torch.arange(nr_classes, device=logits.device)
                    .unsqueeze(0)
                    .expand(*class_probabilities.shape)
                    == label2
                )
                # Probability of correct class
                prediction = class_probabilities[mask].unsqueeze(-1)  # (bs, 1)
                return prediction

    def forward(self, tokens, lengths, label=None, output_attentions=None):
        # https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
        if output_attentions is None:
            output_attentions = []

        output_dict = {}

        # input_ids = tokens["tokens"]["token_ids"] # (bs, seq_len)
        # attention_mask = util.get_text_field_mask(tokens) # (bs, seq_len)

        # Create padding mask
        # http://docs.allennlp.org/v0.9.0/api/allennlp.nn.util.html#allennlp.nn.util.get_text_field_mask
        # 0 where padding, 1 otherwise
        pad_idx = self.vocab.get_padding_index()
        attention_mask = (tokens != pad_idx).bool()  # Orig impl was .long()

        # print(tokens.max(), tokens.min())
        # print(tokens.shape)
        # print(attention_mask.sum())
        # print(type(self.embeddings), self.embeddings)
        # print(self.embeddings(tokens))
        embedding_output = self.embeddings(tokens)  # (bs, seq_len, dim)

        output_dict["embeddings"] = self.embeddings.word_embeddings(tokens).transpose(
            0, 1
        )

        prediction = self.forward_inner(
            embedded_tokens=embedding_output,
            attention_mask=attention_mask,
            label=label,
            output_attentions=output_attentions,
            output_dict=output_dict,
        )

        if prediction is not None:
            output_dict["prediction"] = prediction

        if label is not None:
            output_dict["actual"] = label
            loss = self.loss(
                output_dict["logits"].view(-1, self.num_labels), label.view(-1)
            )
            output_dict["loss"] = loss
            self.metrics["accuracy"](output_dict["class_probabilities"], label)

        return output_dict["logits"], output_dict

    def forward_on_instances(
        self, instances: List[Instance], **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        # An exact copy of the original method, but supports kwargs
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self(**model_input, **kwargs))
            instance_separated_output: List[Dict[str, np.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(
                    instance_separated_output, output
                ):
                    instance_output[name] = batch_element
            return instance_separated_output

    def forward_on_instance(
        self, instance: Instance, **kwargs
    ) -> Dict[str, np.ndarray]:
        return self.forward_on_instances([instance], **kwargs)[0]

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a `label` key to the dictionary with the result.
        """
        output_dict["label"] = torch.argmax(output_dict["class_probabilities"], dim=1)
        return output_dict

    def captum_sub_model(self):
        return _CaptumSubModel(self)

    def instances_to_captum_inputs(self, inputs, lengths, labels=None):
        with torch.no_grad():
            embedded_tokens = self.embeddings(inputs)
            output_dict = {}
            output_dict["embedding"] = embedded_tokens
            _, T = inputs.shape  # get max T
            attention_mask = (
                torch.arange(0, T, device=inputs.device)[None, :] < lengths[:, None]
            )
            return embedded_tokens, None, (attention_mask, labels, output_dict)


class _CaptumSubModel(torch.nn.Module):
    def __init__(self, model: DistilBertForSequenceClassification):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        # (embedded_tokens, attention_mask, label, output_dict)
        inputs_no_attention = inputs[:3] + (None,) + inputs[3:]
        return self.model.forward_inner(*inputs_no_attention)
