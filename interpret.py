import numpy as np

import torch
import torch.nn as nn

from model import JWAttentionClassifier

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
    def __init__(self, name, model, mask_features_by_token=False, attribute_args={}):
        self.attribute_args = attribute_args
        self.mask_features_by_token = mask_features_by_token
        self.predictor = model
        self.name = name

    def interpret_instance(self, instance, lengths, **kwargs):
        # Determine and set additional kwargs for the attribute method

        captum_inputs = self.predictor.instances_to_captum_inputs(instance, lengths)

        # 1. Prepare arguments
        args = dict(
                **kwargs, # Call-based extra arguments
                **self.attribute_kwargs(captum_inputs, mask_features_by_token=self.mask_features_by_token), # General extra arguments
                **self.attribute_args # To be added in subclass constructor
            )
        with torch.no_grad():
            attributions = self.attribute(**args)
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
        vocab = self.predictor.vocab

        # Manually check for distilbert.
        if isinstance(self.predictor, JWAttentionClassifier):
            embedding = self.predictor.embedding
        else: # DistillBert?
            embedding = self.predictor.embeddings # Need to assure the embedding is always fetchable
    
        # Will only work on single-sentence input data
        pad_idx = vocab.get_padding_index()
        pad_idxs = torch.full(inputs.shape[:2], fill_value=pad_idx, device=inputs.device)
        baselines = embedding(pad_idxs)
        print(baselines.shape, inputs.shape)

        attr_kwargs = {
            'inputs' : inputs,
            'target': target,
            'baselines' : baselines,
            'additional_forward_args' : None # set to additional
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
        self.submodel = model.captum_sub_model()
        DeepLift.__init__(self, self.submodel)

# DeepLiftShap
class DeepLiftShapInterpreter(Interpreter, DeepLiftShap):
    def __init__(self, model):
        Interpreter.__init__(self, 'DeepLiftShap', model)
        self.submodel = model.captum_sub_model()
        DeepLift.__init__(self, self.submodel)


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


def legacy_interpret(model, meta):
      sample_sentence = "this is a very nice movie".split()
      sample_instance = torch.tensor(meta.vocab.numericalize(sample_sentence))
      sample_instance = sample_instance.to(device)

      # Try out various interpretability methods
      # attributions = interpret_instance_lime(model, sample_instance)

      # Layer integrated gradients
      # attributions, prediction, delta = interpret_instance_lig(model, sample_instance)

      # Deeplift
      attributions, prediction, delta = interpret_instance_deeplift(model, sample_instance)

      print(attributions.shape) # B, T, E
      attributions = attributions.sum(dim=2).squeeze(0)
      attributions = attributions / torch.norm(attributions)
      attributions = attributions.cpu().detach().numpy()

      visualize_attributions([
          (attributions,
            prediction,
            str(round(prediction)),
            str(round(prediction)),
            "Pos",
            attributions.sum(),
            sample_sentence,
            delta)
        ])