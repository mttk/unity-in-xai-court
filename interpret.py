import numpy as np

import torch
import torch.nn as nn

from captum.attr import visualization

from captum.attr import (
    GradientShap,
    Lime,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

from captum._utils.models.linear_model import SkLearnRidge

class Interpreter:
    # >> Adapted from court-of-xai codebase
    def __init__(self, name, model, mask_features_by_token=False, attribute_args=None):
        self.attribute_args = attribute_args
        self.mask_features_by_token = mask_features_by_token
        self.model = model
        self.name = name

    def interpret_instance(self, instance, **kwargs):
        # Determine and set additional kwargs for the attribute method

        captum_inputs = self.model.instances_to_captum_inputs()

        # 1. Prepare arguments
        args = dict(
                **kwargs, # Call-based extra arguments
                **self.attribute_kwargs(mask_features_by_token=self.mask_features_by_token), # General extra arguments
                self.attribute_args # To be added in subclass constructor
            )
        with torch.inference_mode():
            # 2. Embed instance
            embedded_instance = self.model.embed(instance)

            attributions = self.attribute(embedded_instance)
        return attributions

    def attribute_kwargs(self, captum_inputs, mask_features_by_token=False):
        """
        Args:
            captum_inputs (Tuple): result of model.instances_to_captum_inputs.
            mask_features_by_token (bool, optional): For Captum methods that require a feature mask,
                                                     define each token as a feature if True. If False,
                                                     define each scalar in the embedding dimension as a
                                                     feature (e.g., default behavior in LIME).
                                                     Defaults to False.

        Returns:
            Dict: key-word arguments to be given to the attribute method of the
                  relevant Captum Attribution sub-class.
        """
        inputs, target, additional = captum_inputs
        vocab = self.model.vocab

        # Manually check for distilbert.
        if isinstance(self.predictor._model, DistilBertForSequenceClassification):
            embedding = self.predictor._model.embeddings 
        else:
            embedding = self.model.embedding # Need to assure the embedding is always fetchable
    
        pad_idx = vocab.get_padding_index()
        pad_idx = torch.LongTensor([[pad_idx]]).to(inputs[0].device)
        pad_idxs = tuple(pad_idx.expand(tensor.size()[:2]) for tensor in inputs)
        baselines = tuple(embedding(idx) for idx in pad_idxs)

        attr_kwargs = {
            'inputs' : inputs,
            'target': target,
            'baselines' : baselines,
            'additional_forward_args' : additional
        }

        # For methods that require a feature mask, define each token as one feature
        if mask_features_by_token:
            # see: https://captum.ai/api/lime.html for the definition of a feature mask
            feature_mask_tuple = tuple()
            for i in range(len(inputs)):
                input_tensor = inputs[i]
                bs, seq_len, emb_dim = input_tensor.shape
                feature_mask = torch.tensor(list(range(bs * seq_len))).reshape([bs, seq_len, 1])
                feature_mask = feature_mask.to(inputs[0].device)
                feature_mask = feature_mask.expand(-1, -1, emb_dim)
                feature_mask_tuple += (feature_mask,) # (bs, seq_len, emb_dim)
            attr_kwargs['feature_mask'] = feature_mask_tuple

        return attr_kwargs

# DeepLift
class DeepLiftInterpreter(Interpreter, DeepLift):
    def __init__(self, model):
        Interpreter.__init__(self, 'DeepLift', model)
        self.model = model.captum_sub_model()
        DeepLift.__init__(self, self.model)

# DeepLiftShap
class DeepLiftShapInterpreter(Interpreter, DeepLiftShap):
    def __init__(self, model):
        Interpreter.__init__(self, 'DeepLiftShap', model)
        self.model = model.captum_sub_model()
        DeepLift.__init__(self, self.model)


def visualize_attributions(visualization_records):
    cast_records = []
    for record in visualization_records:
        # Each record is assumed to be a tuple
        print(record)
        cast_records.append(visualization.VisualizationDataRecord(
            *record
            ))
    visualization.visualize_text(cast_records)


def interpret_instance_lime(model, numericalized_instance):
  device = next(iter(model.parameters())).device
  linear_model = SkLearnRidge()
  lime = Lime(model)

  numericalized_instance = numericalized_instance.unsqueeze(0) # Add fake batch dim
  # Feature mask enumerates (word) features in each instance 
  bsz, seq_len = 1, len(numericalized_instance)
  feature_mask = torch.tensor(list(range(bsz*seq_len))).reshape([bsz, seq_len, 1])
  feature_mask = feature_mask.to(device)
  feature_mask = feature_mask.expand(-1, -1, model.embedding_dim)

  attributions = lime.attribute(numericalized_instance,
                                target=1, n_samples=1000,
                                feature_mask=feature_mask) # n samples arg taken from court of xai

  print(attributions.shape)
  print('Lime Attributions:', attributions)
  return attributions

def interpret_instance_deeplift(model, numericalized_instance):

  _model = model.captum_sub_model()
  dl = DeepLift(_model)

  # Should be done in model.prepare_inputs...
  numericalized_instance = numericalized_instance.unsqueeze(0) # Add fake batch dim
  lengths = torch.tensor(len(numericalized_instance)).unsqueeze(0)
  logits, return_dict = model(numericalized_instance, lengths)
  pred = logits.squeeze() # obtain prediction
  scaled_pred = nn.Sigmoid()(pred).item() # scale to probability

  # Reference indices are just a bunch of padding indices
  # token_reference = TokenReferenceBase(reference_token_idx=0) # Padding index is the reference
  # reference_indices = token_reference.generate_reference(len(numericalized_instance), 
  #                                                        device=next(iter(model.parameters())).device).unsqueeze(0)
  with torch.no_grad():
    embedded_instance = model.embedding(numericalized_instance)
    # Pass embeddings to input

    outs, delta = dl.attribute(embedded_instance, return_convergence_delta=True)
  print(outs)
  return outs, scaled_pred, delta


def interpret_instance_lig(model, numericalized_instance):
  _model = model.captum_sub_model()
  lig = LayerIntegratedGradients(_model, model.embedding) # LIG uses embedding data

  numericalized_instance = numericalized_instance.unsqueeze(0) # Add fake batch dim
  lengths = torch.tensor(len(numericalized_instance)).unsqueeze(0)
  logits, return_dict = model(numericalized_instance, lengths)
  pred = logits.squeeze() # obtain prediction
  # print(pred)
  scaled_pred = nn.Sigmoid()(pred).item() # scale to probability

  # Reference indices are just a bunch of padding indices
  token_reference = TokenReferenceBase(reference_token_idx=0) # Padding index is the reference
  reference_indices = token_reference.generate_reference(len(numericalized_instance), 
                                                          device=next(iter(model.parameters())).device).unsqueeze(0)
  with torch.no_grad():
    embedded_instance = model.embedding(numericalized_instance)

    attributions, delta = lig.attribute(embedded_instance, reference_indices,
                                        n_steps=500, return_convergence_delta=True)
  print('IG Attributions:', attributions)
  print('Convergence Delta:', delta)
  return attributions, scaled_pred, delta
